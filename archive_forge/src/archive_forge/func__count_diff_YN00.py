import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _count_diff_YN00(codon1, codon2, P, codons, codon_table):
    """Count differences between two codons (three-letter string; PRIVATE).

    The function will weighted multiple pathways from codon1 to codon2
    according to P matrix of codon substitution. The proportion
    of transition and transversion (TV) will also be calculated in
    the function.
    """
    TV = [0, 0, 0, 0]
    if codon1 == codon2:
        return TV
    else:
        diff_pos = [i for i, (nucleotide1, nucleotide2) in enumerate(zip(codon1, codon2)) if nucleotide1 != nucleotide2]

        def count_TV(codon1, codon2, diff, codon_table, weight=1):
            purine = ('A', 'G')
            pyrimidine = ('T', 'C')
            dic = codon_table.forward_table
            stop = codon_table.stop_codons
            if codon1 in stop or codon2 in stop:
                if codon1[diff] in purine and codon2[diff] in purine:
                    return [0, 0, weight, 0]
                elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                    return [0, 0, weight, 0]
                else:
                    return [0, 0, 0, weight]
            elif dic[codon1] == dic[codon2]:
                if codon1[diff] in purine and codon2[diff] in purine:
                    return [weight, 0, 0, 0]
                elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                    return [weight, 0, 0, 0]
                else:
                    return [0, weight, 0, 0]
            elif codon1[diff] in purine and codon2[diff] in purine:
                return [0, 0, weight, 0]
            elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                return [0, 0, weight, 0]
            else:
                return [0, 0, 0, weight]
        if len(diff_pos) == 1:
            TV = [p + q for p, q in zip(TV, count_TV(codon1, codon2, diff_pos[0], codon_table))]
        elif len(diff_pos) == 2:
            tmp_codons = [codon1[:i] + codon2[i] + codon1[i + 1:] for i in diff_pos]
            path_prob = []
            for codon in tmp_codons:
                codon_idx = list(map(codons.index, [codon1, codon, codon2]))
                prob = (P[codon_idx[0], codon_idx[1]], P[codon_idx[1], codon_idx[2]])
                path_prob.append(prob[0] * prob[1])
            path_prob = [2 * i / sum(path_prob) for i in path_prob]
            for n, i in enumerate(diff_pos):
                codon = codon1[:i] + codon2[i] + codon1[i + 1:]
                TV = [p + q for p, q in zip(TV, count_TV(codon1, codon, i, codon_table, weight=path_prob[n] / 2))]
                TV = [p + q for p, q in zip(TV, count_TV(codon1, codon, i, codon_table, weight=path_prob[n] / 2))]
        elif len(diff_pos) == 3:
            paths = list(permutations([0, 1, 2], 3))
            path_prob = []
            tmp_codons = []
            for index1, index2, index3 in paths:
                tmp1 = codon1[:index1] + codon2[index1] + codon1[index1 + 1:]
                tmp2 = tmp1[:index2] + codon2[index2] + tmp1[index2 + 1:]
                tmp_codons.append((tmp1, tmp2))
                codon_idx = list(map(codons.index, [codon1, tmp1, tmp2, codon2]))
                prob = (P[codon_idx[0], codon_idx[1]], P[codon_idx[1], codon_idx[2]], P[codon_idx[2], codon_idx[3]])
                path_prob.append(prob[0] * prob[1] * prob[2])
            path_prob = [3 * i / sum(path_prob) for i in path_prob]
            for codon, j, k in zip(tmp_codons, path_prob, paths):
                TV = [p + q for p, q in zip(TV, count_TV(codon1, codon[0], k[0], codon_table, weight=j / 3))]
                TV = [p + q for p, q in zip(TV, count_TV(codon[0], codon[1], k[1], codon_table, weight=j / 3))]
                TV = [p + q for p, q in zip(TV, count_TV(codon[1], codon2, k[1], codon_table, weight=j / 3))]
    return TV