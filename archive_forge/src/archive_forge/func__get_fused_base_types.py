from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _get_fused_base_types(self, fused_compound_types):
    """
        Get a list of unique basic fused types, from a list of
        (possibly) compound fused types.
        """
    base_types = []
    seen = set()
    for fused_type in fused_compound_types:
        fused_type.get_fused_types(result=base_types, seen=seen)
    return base_types