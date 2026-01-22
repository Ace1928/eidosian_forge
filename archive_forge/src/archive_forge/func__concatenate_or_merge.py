from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
def _concatenate_or_merge(partition_1, partition_2, x, i, ref_weight):
    ccx = _pivot(partition_1, x)
    cci = _pivot(partition_2, i)
    merged_xi = ccx.union(cci)
    if _weight_of_cluster(frozenset(merged_xi)) <= ref_weight:
        cp1 = list(filter(lambda x: x != ccx, partition_1))
        cp2 = list(filter(lambda x: x != cci, partition_2))
        option_2 = [merged_xi] + cp1 + cp2
        return (option_2, _value_of_partition(option_2))
    else:
        option_1 = partition_1 + partition_2
        return (option_1, _value_of_partition(option_1))