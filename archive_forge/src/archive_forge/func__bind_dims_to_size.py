import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _bind_dims_to_size(lhs_size, rhs, lhs_debug):
    from . import DimensionMismatchError
    not_bound = tuple(((i, r) for i, r in enumerate(rhs) if not r.is_bound))
    if len(not_bound) == 1:
        idx, d = not_bound[0]
        rhs_so_far = prod((r.size for r in rhs if r.is_bound))
        if lhs_size % rhs_so_far != 0:
            rhs_s = tuple(('?' if not r.is_bound else str(r.size) for r in rhs))
            raise DimensionMismatchError(f'inferred dimension does not evenly fit into larger dimension: {lhs_size} vs {rhs_s}')
        new_size = lhs_size // rhs_so_far
        d.size = new_size
    elif len(not_bound) > 1:
        rhs_s = tuple(('?' if not r.is_bound else str(r.size) for r in rhs))
        raise DimensionMismatchError(f'cannot infer the size of two dimensions at once: {rhs} with sizes {rhs_s}')
    else:
        rhs_size = prod((r.size for r in rhs))
        if lhs_size != rhs_size:
            raise DimensionMismatchError(f'Dimension sizes to do not match ({lhs_size} != {rhs_size}) when matching {lhs_debug} to {rhs}')