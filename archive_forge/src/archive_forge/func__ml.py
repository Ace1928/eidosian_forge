import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _ml(codons1, codons2, cmethod, codon_table):
    """ML method main function (PRIVATE)."""
    from scipy.optimize import minimize
    pi = _get_pi(codons1, codons2, cmethod, codon_table=codon_table)
    codon_cnt = Counter(zip(codons1, codons2))
    codons = [codon for codon in list(codon_table.forward_table.keys()) + codon_table.stop_codons if 'U' not in codon]

    def func(params, pi=pi, codon_cnt=codon_cnt, codons=codons, codon_table=codon_table):
        """Temporary function, params = [t, k, w]."""
        return -_likelihood_func(params[0], params[1], params[2], pi, codon_cnt, codons=codons, codon_table=codon_table)
    opt_res = minimize(func, [1, 0.1, 2], method='L-BFGS-B', bounds=((1e-10, 20), (1e-10, 20), (1e-10, 10)), tol=1e-05)
    t, k, w = opt_res.x
    Q = _get_Q(pi, k, w, codons, codon_table)
    Sd = Nd = 0
    for i, codon1 in enumerate(codons):
        for j, codon2 in enumerate(codons):
            if i != j:
                try:
                    if codon_table.forward_table[codon1] == codon_table.forward_table[codon2]:
                        Sd += pi[codon1] * Q[i, j]
                    else:
                        Nd += pi[codon1] * Q[i, j]
                except KeyError:
                    pass
    Sd *= t
    Nd *= t

    def func_w1(params, pi=pi, codon_cnt=codon_cnt, codons=codons, codon_table=codon_table):
        """Temporary function, params = [t, k]. w is fixed to 1."""
        return -_likelihood_func(params[0], params[1], 1.0, pi, codon_cnt, codons=codons, codon_table=codon_table)
    opt_res = minimize(func_w1, [1, 0.1], method='L-BFGS-B', bounds=((1e-10, 20), (1e-10, 20)), tol=1e-05)
    t, k = opt_res.x
    w = 1.0
    Q = _get_Q(pi, k, w, codons, codon_table)
    rhoS = rhoN = 0
    for i, codon1 in enumerate(codons):
        for j, codon2 in enumerate(codons):
            if i != j:
                try:
                    if codon_table.forward_table[codon1] == codon_table.forward_table[codon2]:
                        rhoS += pi[codon1] * Q[i, j]
                    else:
                        rhoN += pi[codon1] * Q[i, j]
                except KeyError:
                    pass
    rhoS *= 3
    rhoN *= 3
    dN = Nd / rhoN
    dS = Sd / rhoS
    return (dN, dS)