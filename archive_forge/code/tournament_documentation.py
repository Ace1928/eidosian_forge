from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
Decides whether the given set of nodes is closed.

        A set *S* of nodes is *closed* if for each node *u* in the graph
        not in *S* and for each node *v* in *S*, there is an edge from
        *u* to *v*.

        