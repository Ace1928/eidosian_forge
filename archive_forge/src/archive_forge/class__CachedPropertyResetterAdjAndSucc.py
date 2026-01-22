from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
class _CachedPropertyResetterAdjAndSucc:
    """Data Descriptor class that syncs and resets cached properties adj and succ

    The cached properties `adj` and `succ` are reset whenever `_adj` or `_succ`
    are set to new objects. In addition, the attributes `_succ` and `_adj`
    are synced so these two names point to the same object.

    This object sits on a class and ensures that any instance of that
    class clears its cached properties "succ" and "adj" whenever the
    underlying instance attributes "_succ" or "_adj" are set to a new object.
    It only affects the set process of the obj._adj and obj._succ attribute.
    All get/del operations act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od['_adj'] = value
        od['_succ'] = value
        if 'adj' in od:
            del od['adj']
        if 'succ' in od:
            del od['succ']