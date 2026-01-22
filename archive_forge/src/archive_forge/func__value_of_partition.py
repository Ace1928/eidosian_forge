from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
def _value_of_partition(partition):
    return sum((_value_of_cluster(frozenset(c)) for c in partition))