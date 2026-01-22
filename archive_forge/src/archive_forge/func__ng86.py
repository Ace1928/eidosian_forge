import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _ng86(codons1, codons2, k, codon_table):
    """NG86 method main function (PRIVATE)."""
    S_sites1, N_sites1 = _count_site_NG86(codons1, codon_table=codon_table, k=k)
    S_sites2, N_sites2 = _count_site_NG86(codons2, codon_table=codon_table, k=k)
    S_sites = (S_sites1 + S_sites2) / 2.0
    N_sites = (N_sites1 + N_sites2) / 2.0
    SN = [0, 0]
    for codon1, codon2 in zip(codons1, codons2):
        SN = [m + n for m, n in zip(SN, _count_diff_NG86(codon1, codon2, codon_table=codon_table))]
    ps = SN[0] / S_sites
    pn = SN[1] / N_sites
    if ps < 3 / 4:
        dS = abs(-3.0 / 4 * log(1 - 4.0 / 3 * ps))
    else:
        dS = -1
    if pn < 3 / 4:
        dN = abs(-3.0 / 4 * log(1 - 4.0 / 3 * pn))
    else:
        dN = -1
    return (dN, dS)