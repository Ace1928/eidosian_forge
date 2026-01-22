from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
class _CachedPropertyResetterPred:
    """Data Descriptor class for _pred that resets ``pred`` cached_property when needed

    This assumes that the ``cached_property`` ``G.pred`` should be reset whenever
    ``G._pred`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "pred" whenever the underlying
    instance attribute "_pred" is set to a new object. It only affects
    the set process of the obj._pred attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od['_pred'] = value
        if 'pred' in od:
            del od['pred']