import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _augment_blast_hsp(hsp, attr):
    """Calculate the given HSP attribute, for writing (PRIVATE)."""
    if not hasattr(hsp, attr) and (not attr.endswith('_pct')):
        if attr == 'aln_span':
            hsp.aln_span = hsp.ident_num + hsp.mismatch_num + hsp.gap_num
        elif attr.startswith('ident'):
            setattr(hsp, attr, hsp.aln_span - hsp.mismatch_num - hsp.gap_num)
        elif attr.startswith('gap'):
            setattr(hsp, attr, hsp.aln_span - hsp.ident_num - hsp.mismatch_num)
        elif attr == 'mismatch_num':
            setattr(hsp, attr, hsp.aln_span - hsp.ident_num - hsp.gap_num)
        elif attr == 'gapopen_num':
            if not hasattr(hsp, 'query') or not hasattr(hsp, 'hit'):
                raise AttributeError
            hsp.gapopen_num = _compute_gapopen_num(hsp)
    if attr == 'ident_pct':
        hsp.ident_pct = hsp.ident_num / hsp.aln_span * 100
    elif attr == 'pos_pct':
        hsp.pos_pct = hsp.pos_num / hsp.aln_span * 100
    elif attr == 'gap_pct':
        hsp.gap_pct = hsp.gap_num / hsp.aln_span * 100