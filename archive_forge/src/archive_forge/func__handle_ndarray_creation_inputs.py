from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable
@classmethod
def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
    from sympy.matrices.matrices import MatrixBase
    from sympy.tensor.array import SparseNDimArray
    if shape is None:
        if iterable is None:
            shape = ()
            iterable = ()
        elif isinstance(iterable, SparseNDimArray):
            return (iterable._shape, iterable._sparse_array)
        elif isinstance(iterable, NDimArray):
            shape = iterable.shape
        elif isinstance(iterable, Iterable):
            iterable, shape = cls._scan_iterable_shape(iterable)
        elif isinstance(iterable, MatrixBase):
            shape = iterable.shape
        else:
            shape = ()
            iterable = (iterable,)
    if isinstance(iterable, (Dict, dict)) and shape is not None:
        new_dict = iterable.copy()
        for k, v in new_dict.items():
            if isinstance(k, (tuple, Tuple)):
                new_key = 0
                for i, idx in enumerate(k):
                    new_key = new_key * shape[i] + idx
                iterable[new_key] = iterable[k]
                del iterable[k]
    if isinstance(shape, (SYMPY_INTS, Integer)):
        shape = (shape,)
    if not all((isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape)):
        raise TypeError('Shape should contain integers only.')
    return (tuple(shape), iterable)