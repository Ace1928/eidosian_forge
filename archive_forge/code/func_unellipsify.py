from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def unellipsify(indices, ndim):
    result = []
    seen_ellipsis = False
    have_slices = False
    newaxes = [newaxis for newaxis in indices if newaxis.is_none]
    n_indices = len(indices) - len(newaxes)
    for index in indices:
        if isinstance(index, ExprNodes.EllipsisNode):
            have_slices = True
            full_slice = empty_slice(index.pos)
            if seen_ellipsis:
                result.append(full_slice)
            else:
                nslices = ndim - n_indices + 1
                result.extend([full_slice] * nslices)
                seen_ellipsis = True
        else:
            have_slices = have_slices or index.is_slice or index.is_none
            result.append(index)
    result_length = len(result) - len(newaxes)
    if result_length < ndim:
        have_slices = True
        nslices = ndim - result_length
        result.extend([empty_slice(indices[-1].pos)] * nslices)
    return (have_slices, result, newaxes)