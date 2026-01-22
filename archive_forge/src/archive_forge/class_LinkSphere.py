import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
class LinkSphere(LinkSurface):
    """
    >>> T = Mcomplex('kLLLLQMkbcghgihijjjtsmnonnkddl')  # m004(1, 2)
    >>> L = LinkSphere(T)
    >>> L.edge_graph().number_of_nodes()
    22
    >>> 2 * len(T.Edges)
    22
    """

    def __init__(self, t3m_triangulation):
        N = t3m_triangulation
        assert len(N.Vertices) == 1 and N.Vertices[0].link_genus() == 0
        LinkSurface.__init__(self, N)