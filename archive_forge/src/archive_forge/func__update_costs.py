from itertools import count
import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import BinaryHeap, not_implemented_for, py_random_state
def _update_costs(costs_x, x):
    for y, w in edges[x]:
        costs_y = costs[side[y]]
        cost_y = costs_y.get(y)
        if cost_y is not None:
            cost_y += 2 * (-w if costs_x is costs_y else w)
            costs_y.insert(y, cost_y, True)