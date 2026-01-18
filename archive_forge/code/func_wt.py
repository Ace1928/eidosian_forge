from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
def wt(u, v):
    return G[u][v].get(weight, 1) / max_weight