import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
def _swap_node_partition(cut, node):
    return cut - {node} if node in cut else cut.union({node})