import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _calc_millibad(psl, is_protein):
    """Calculate millibad (PRIVATE)."""
    size_mul = 3 if is_protein else 1
    millibad = 0
    qali_size = size_mul * (psl['qend'] - psl['qstart'])
    tali_size = psl['tend'] - psl['tstart']
    ali_size = min(qali_size, tali_size)
    if ali_size <= 0:
        return 0
    size_dif = qali_size - tali_size
    size_dif = 0 if size_dif < 0 else size_dif
    total = size_mul * (psl['matches'] + psl['repmatches'] + psl['mismatches'])
    if total != 0:
        millibad = 1000 * (psl['mismatches'] * size_mul + psl['qnuminsert'] + round(3 * log(1 + size_dif))) / total
    return millibad