import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
class _BaseExonerateParser(ABC):
    """Abstract base class iterator for exonerate format."""
    _ALN_MARK: Optional[str] = None

    def __init__(self, handle):
        self.handle = handle
        self.has_c4_alignment = False

    def __iter__(self):
        while True:
            self.line = self.handle.readline()
            if self.line.startswith('C4 Alignment:') and (not self.has_c4_alignment):
                self.has_c4_alignment = True
            if self.line.startswith('C4 Alignment:') or self.line.startswith('vulgar:') or self.line.startswith('cigar:'):
                break
            elif not self.line or self.line.startswith('-- completed '):
                return
        for qresult in self._parse_qresult():
            qresult.program = 'exonerate'
            qresult.description = qresult.description
            for hit in qresult:
                hit.description = hit.description
            yield qresult

    def read_until(self, bool_func):
        """Read the file handle until the given bool function returns True."""
        while True:
            if not self.line or bool_func(self.line):
                return
            else:
                self.line = self.handle.readline()

    @abstractmethod
    def parse_alignment_block(self, header):
        raise NotImplementedError

    def _parse_alignment_header(self):
        aln_header = []
        while self.line.strip():
            aln_header.append(self.line.strip())
            self.line = self.handle.readline()
        qresult, hit, hsp = ({}, {}, {})
        for line in aln_header:
            if line.startswith('Query:'):
                qresult['id'], qresult['description'] = _parse_hit_or_query_line(line)
            elif line.startswith('Target:'):
                hit['id'], hit['description'] = _parse_hit_or_query_line(line)
            elif line.startswith('Model:'):
                qresult['model'] = line.split(' ', 1)[1]
            elif line.startswith('Raw score:'):
                hsp['score'] = line.split(' ', 2)[2]
            elif line.startswith('Query range:'):
                hsp['query_start'], hsp['query_end'] = line.split(' ', 4)[2:5:2]
            elif line.startswith('Target range:'):
                hsp['hit_start'], hsp['hit_end'] = line.split(' ', 4)[2:5:2]
        qresult_strand, qresult_desc = _get_strand_from_desc(desc=qresult['description'], is_protein='protein2' in qresult['model'], modify_desc=True)
        hsp['query_strand'] = qresult_strand
        qresult['description'] = qresult_desc
        hit_strand, hit_desc = _get_strand_from_desc(desc=hit['description'], is_protein='2protein' in qresult['model'], modify_desc=True)
        hsp['hit_strand'] = hit_strand
        hit['description'] = hit_desc
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp}

    def _parse_qresult(self):
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        state_HIT_NEW = 2
        state_HIT_SAME = 4
        qres_state, hit_state = (None, None)
        file_state = None
        cur_qid, cur_hid = (None, None)
        prev_qid, prev_hid = (None, None)
        cur, prev = (None, None)
        hit_list, hsp_list = ([], [])
        if self.has_c4_alignment:
            self._ALN_MARK = 'C4 Alignment:'
        while True:
            self.read_until(lambda line: line.startswith(self._ALN_MARK))
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
                prev_hid = cur_hid
            if self.line:
                assert self.line.startswith(self._ALN_MARK), self.line
                header = {'qresult': {}, 'hit': {}, 'hsp': {}}
                if self.has_c4_alignment:
                    self.read_until(lambda line: line.strip().startswith('Query:'))
                    header = self._parse_alignment_header()
                cur = self.parse_alignment_block(header)
                cur_qid = cur['qresult']['id']
                cur_hid = cur['hit']['id']
            elif not self.line or self.line.startswith('-- completed '):
                file_state = state_EOF
                cur_qid, cur_hid = (None, None)
            if prev_qid != cur_qid:
                qres_state = state_QRES_NEW
            else:
                qres_state = state_QRES_SAME
            if prev_hid != cur_hid or qres_state == state_QRES_NEW:
                hit_state = state_HIT_NEW
            else:
                hit_state = state_HIT_SAME
            if prev is not None:
                hsp = _create_hsp(prev_hid, prev_qid, prev['hsp'])
                hsp_list.append(hsp)
                if hit_state == state_HIT_NEW:
                    hit = Hit(hsp_list)
                    for attr, value in prev['hit'].items():
                        setattr(hit, attr, value)
                    hit_list.append(hit)
                    hsp_list = []
                if qres_state == state_QRES_NEW or file_state == state_EOF:
                    qresult = QueryResult(id=prev_qid)
                    for hit in hit_list:
                        qresult.absorb(hit)
                    for attr, value in prev['qresult'].items():
                        setattr(qresult, attr, value)
                    yield qresult
                    if file_state == state_EOF:
                        break
                    hit_list = []
            if not self.has_c4_alignment:
                self.line = self.handle.readline()