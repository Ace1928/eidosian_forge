import networkx as nx
from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, PlanarEmbedding
from networkx.classes.reportviews import NodeView
class LoopbackMultiGraph(MultiGraph):
    __networkx_backend__ = 'nx-loopback'