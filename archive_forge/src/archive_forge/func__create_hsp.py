import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _create_hsp(hid, qid, hspd):
    """Return a list of HSP objects from the given parsed HSP values (PRIVATE)."""
    frags = []
    for idx, qcoords in enumerate(hspd['query_ranges']):
        hseqlist = hspd.get('hit')
        hseq = '' if hseqlist is None else hseqlist[idx]
        qseqlist = hspd.get('query')
        qseq = '' if qseqlist is None else qseqlist[idx]
        frag = HSPFragment(hid, qid, hit=hseq, query=qseq)
        frag.query_start = qcoords[0]
        frag.query_end = qcoords[1]
        frag.hit_start = hspd['hit_ranges'][idx][0]
        frag.hit_end = hspd['hit_ranges'][idx][1]
        try:
            aln_annot = hspd.get('aln_annotation', {})
            for key, value in aln_annot.items():
                frag.aln_annotation[key] = value[idx]
        except IndexError:
            pass
        frag.query_strand = hspd['query_strand']
        frag.hit_strand = hspd['hit_strand']
        if frag.aln_annotation.get('similarity') is not None:
            if '#' in frag.aln_annotation['similarity']:
                frags.extend(_split_fragment(frag))
                continue
        if len(frag.aln_annotation) > 1 or frag.query_strand == 0 or ('vulgar_comp' in hspd and re.search(_RE_TRANS, hspd['vulgar_comp'])):
            _set_frame(frag)
        frags.append(frag)
    if len(frags[0].aln_annotation) == 2:
        frags = _adjust_aa_seq(frags)
    hsp = HSP(frags)
    for attr in ('score', 'hit_split_codons', 'query_split_codons', 'model', 'vulgar_comp', 'cigar_comp', 'molecule_type'):
        if attr in hspd:
            setattr(hsp, attr, hspd[attr])
    return hsp