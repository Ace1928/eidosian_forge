from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
class _CachedPropertyResetterNode:
    """Data Descriptor class for _node that resets ``nodes`` cached_property when needed

    This assumes that the ``cached_property`` ``G.node`` should be reset whenever
    ``G._node`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "nodes" whenever the underlying
    instance attribute "_node" is set to a new object. It only affects
    the set process of the obj._adj attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od['_node'] = value
        if 'nodes' in od:
            del od['nodes']