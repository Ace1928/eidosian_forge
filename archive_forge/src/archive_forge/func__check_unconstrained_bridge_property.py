import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def _check_unconstrained_bridge_property(G, info1):
    import math
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    C = collapse(G, bridge_ccs)
    p = len([n for n, d in C.degree() if d == 1])
    q = len([n for n, d in C.degree() if d == 0])
    if p + q > 1:
        size_target = math.ceil(p / 2) + q
        size_aug = info1['num_edges']
        assert size_aug == size_target, 'augmentation size is different from what theory predicts'