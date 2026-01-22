import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
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