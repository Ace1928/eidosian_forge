from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def to_undirected_class(self):
    """Returns the class to use for empty undirected copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
    return Graph