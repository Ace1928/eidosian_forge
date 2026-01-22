import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
def _coo_gen_triples(A):
    """Converts a SciPy sparse array in **Coordinate** format to an iterable
    of weighted edge triples.

    """
    return ((int(i), int(j), d) for i, j, d in zip(A.row, A.col, A.data))