import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _count_site_NG86(codons, codon_table, k=1):
    """Count synonymous and non-synonymous sites of a list of codons (PRIVATE).

    Arguments:
     - codons - A list of three letter codons.
     - k - transition/transversion rate ratio.

    """
    S_site = 0
    N_site = 0
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    bases = ('A', 'T', 'C', 'G')
    for codon in codons:
        neighbor_codon = {'transition': [], 'transversion': []}
        codon = codon.replace('U', 'T')
        for i, nucleotide in enumerate(codon):
            for base in bases:
                if nucleotide == base:
                    pass
                elif nucleotide in purine and base in purine:
                    codon_chars = list(codon)
                    codon_chars[i] = base
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transition'].append(this_codon)
                elif nucleotide in pyrimidine and base in pyrimidine:
                    codon_chars = list(codon)
                    codon_chars[i] = base
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transition'].append(this_codon)
                else:
                    codon_chars = list(codon)
                    codon_chars[i] = base
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transversion'].append(this_codon)
        aa = codon_table.forward_table[codon]
        this_codon_N_site = this_codon_S_site = 0
        for neighbor in neighbor_codon['transition']:
            if neighbor in codon_table.stop_codons:
                this_codon_N_site += 1
            elif codon_table.forward_table[neighbor] == aa:
                this_codon_S_site += 1
            else:
                this_codon_N_site += 1
        for neighbor in neighbor_codon['transversion']:
            if neighbor in codon_table.stop_codons:
                this_codon_N_site += k
            elif codon_table.forward_table[neighbor] == aa:
                this_codon_S_site += k
            else:
                this_codon_N_site += k
        norm_const = (this_codon_N_site + this_codon_S_site) / 3
        S_site += this_codon_S_site / norm_const
        N_site += this_codon_N_site / norm_const
    return (S_site, N_site)