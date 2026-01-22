import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _determine_from_signature(self, signature):
    if isinstance(signature, tuple):
        if len(signature) == 1:
            raise TypeError('The use of a length 1 tuple for the ufunc `signature` is not allowed. Use `dtype` or  fill the tuple with `None`s.')
        nin = self._nin
        nout = self._nout
        if len(signature) != nin + nout:
            raise TypeError(f'A type-tuple must be specified of length 1 or 3 for ufunc {self._name}')
        signature = ''.join((numpy.dtype(t).char for t in signature[:nin])) + '->' + ''.join((numpy.dtype(t).char for t in signature[nin:nin + nout]))
    if isinstance(signature, str):
        is_out = len(signature) == 1
        for op in self._ops:
            if is_out:
                for t in op.out_types:
                    if t.char != signature:
                        break
                else:
                    return op
            elif op.sig_str == signature:
                return op
    raise TypeError(f'No loop matching the specified signature and casting was found for ufunc {self._name}')