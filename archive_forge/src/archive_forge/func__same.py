import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def _same(measure, tol=0):
    vals = set(measure.values())
    if max(vals) - min(vals) <= tol:
        return True
    return False