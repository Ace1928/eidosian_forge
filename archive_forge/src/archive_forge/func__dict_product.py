from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _dict_product(d1, d2):
    return {k: (d1.get(k), d2.get(k)) for k in set(d1) | set(d2)}