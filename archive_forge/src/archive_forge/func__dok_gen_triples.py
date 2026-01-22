import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
def _dok_gen_triples(A):
    """Converts a SciPy sparse array in **Dictionary of Keys** format to an
    iterable of weighted edge triples.

    """
    for (r, c), v in A.items():
        yield (r, c, v)