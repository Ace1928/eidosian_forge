from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
@lru_cache(CLUSTER_EVAL_CACHE_SIZE)
def _weight_of_cluster(cluster):
    return sum((safe_G.nodes[n][node_weight] for n in cluster))