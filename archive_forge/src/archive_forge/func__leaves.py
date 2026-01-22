from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('undirected')
def _leaves(gr):
    for x in gr.nodes:
        if not nx.descendants(gr, x):
            yield x