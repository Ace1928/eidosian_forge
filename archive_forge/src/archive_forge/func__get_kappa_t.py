import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_kappa_t(pi, TV, t=False):
    """Calculate kappa (PRIVATE).

    The following formula and variable names are according to PMID: 10666704
    """
    pi['Y'] = pi['T'] + pi['C']
    pi['R'] = pi['A'] + pi['G']
    A = (2 * (pi['T'] * pi['C'] + pi['A'] * pi['G']) + 2 * (pi['T'] * pi['C'] * pi['R'] / pi['Y'] + pi['A'] * pi['G'] * pi['Y'] / pi['R']) * (1 - TV[1] / (2 * pi['Y'] * pi['R'])) - TV[0]) / (2 * (pi['T'] * pi['C'] / pi['Y'] + pi['A'] * pi['G'] / pi['R']))
    B = 1 - TV[1] / (2 * pi['Y'] * pi['R'])
    a = -0.5 * log(A)
    b = -0.5 * log(B)
    kappaF84 = a / b - 1
    if t is False:
        kappaHKY85 = 1 + (pi['T'] * pi['C'] / pi['Y'] + pi['A'] * pi['G'] / pi['R']) * kappaF84 / (pi['T'] * pi['C'] + pi['A'] * pi['G'])
        return kappaHKY85
    else:
        t = (4 * pi['T'] * pi['C'] * (1 + kappaF84 / pi['Y']) + 4 * pi['A'] * pi['G'] * (1 + kappaF84 / pi['R']) + 4 * pi['Y'] * pi['R']) * b
        return t