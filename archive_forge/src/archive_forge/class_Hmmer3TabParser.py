from itertools import chain
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class Hmmer3TabParser:
    """Parser for the HMMER table format."""

    def __init__(self, handle):
        """Initialize the class."""
        self.handle = handle
        self.line = self.handle.readline()

    def __iter__(self):
        """Iterate over Hmmer3TabParser, yields query results."""
        header_mark = '#'
        while self.line.startswith(header_mark):
            self.line = self.handle.readline()
        if self.line:
            yield from self._parse_qresult()

    def _parse_row(self):
        """Return a dictionary of parsed row values (PRIVATE)."""
        cols = [x for x in self.line.strip().split(' ') if x]
        if len(cols) < 18:
            raise ValueError('Less columns than expected, only %i' % len(cols))
        cols[18] = ' '.join(cols[18:])
        qresult = {}
        qresult['id'] = cols[2]
        qresult['accession'] = cols[3]
        hit = {}
        hit['id'] = cols[0]
        hit['accession'] = cols[1]
        hit['evalue'] = float(cols[4])
        hit['bitscore'] = float(cols[5])
        hit['bias'] = float(cols[6])
        hit['domain_exp_num'] = float(cols[10])
        hit['region_num'] = int(cols[11])
        hit['cluster_num'] = int(cols[12])
        hit['overlap_num'] = int(cols[13])
        hit['env_num'] = int(cols[14])
        hit['domain_obs_num'] = int(cols[15])
        hit['domain_reported_num'] = int(cols[16])
        hit['domain_included_num'] = int(cols[17])
        hit['description'] = cols[18]
        hsp = {}
        hsp['evalue'] = float(cols[7])
        hsp['bitscore'] = float(cols[8])
        hsp['bias'] = float(cols[9])
        frag = {}
        frag['hit_strand'] = frag['query_strand'] = 0
        frag['molecule_type'] = 'protein'
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}

    def _parse_qresult(self):
        """Return QueryResult objects (PRIVATE)."""
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        qres_state = None
        file_state = None
        prev_qid = None
        cur, prev = (None, None)
        hit_list = []
        cur_qid = None
        while True:
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
            if self.line and (not self.line.startswith('#')):
                cur = self._parse_row()
                cur_qid = cur['qresult']['id']
            else:
                file_state = state_EOF
                cur_qid = None
            if prev_qid != cur_qid:
                qres_state = state_QRES_NEW
            else:
                qres_state = state_QRES_SAME
            if prev is not None:
                prev_hid = prev['hit']['id']
                frag = HSPFragment(prev_hid, prev_qid)
                for attr, value in prev['frag'].items():
                    setattr(frag, attr, value)
                hsp = HSP([frag])
                for attr, value in prev['hsp'].items():
                    setattr(hsp, attr, value)
                hit = Hit([hsp])
                for attr, value in prev['hit'].items():
                    setattr(hit, attr, value)
                hit_list.append(hit)
                if qres_state == state_QRES_NEW or file_state == state_EOF:
                    qresult = QueryResult(hit_list, prev_qid)
                    for attr, value in prev['qresult'].items():
                        setattr(qresult, attr, value)
                    yield qresult
                    if file_state == state_EOF:
                        break
                    hit_list = []
            self.line = self.handle.readline()