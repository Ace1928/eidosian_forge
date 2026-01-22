import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _select_k(n, seed):
    r = seed.randint(0, (n + 1) ** (n - 1) - 1)
    cum_sum = 0
    for k in range(1, n):
        cum_sum += factorial(n - 1) * n ** (n - k) // (factorial(k - 1) * factorial(n - k))
        if r < cum_sum:
            return k
    return n