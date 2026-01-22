import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def _unary_operation(self, ufunc, method, *args, **kwargs):
    """Run recursion to perform unary_funcs on BlockVector"""
    x = args[0]
    if isinstance(x, BlockVector):
        v = BlockVector(x.nblocks)
        for i in range(x.nblocks):
            _args = [x.get_block(i)] + [args[j] for j in range(1, len(args))]
            v.set_block(i, self._unary_operation(ufunc, method, *_args, **kwargs))
        return v
    elif type(x) == np.ndarray:
        return super(BlockVector, self).__array_ufunc__(ufunc, method, *args, **kwargs)
    else:
        raise NotImplementedError()