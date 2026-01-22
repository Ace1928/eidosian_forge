import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _calc_score(psl, is_protein):
    """Calculate score (PRIVATE)."""
    size_mul = 3 if is_protein else 1
    return size_mul * (psl['matches'] + (psl['repmatches'] >> 1)) - size_mul * psl['mismatches'] - psl['qnuminsert'] - psl['tnuminsert']