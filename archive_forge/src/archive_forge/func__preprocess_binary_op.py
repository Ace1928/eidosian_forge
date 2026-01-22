from inspect import signature
from math import prod
import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype
import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage
from .utils import try_convert_from_interoperable_type
def _preprocess_binary_op(self, other, cast_input_types=True, dtype=None, out=None):
    """
        Processes arguments and performs dtype conversions necessary to perform binary
        operations. If the arguments to the binary operation are a 1D object and a 2D object,
        then it will swap the order of the caller and callee return values in order to
        facilitate native broadcasting by modin.

        This function may modify `self._query_compiler` and `other._query_compiler` by replacing
        it with the result of `astype`.

        Parameters
        ----------
        other : array or scalar
            The RHS of the binary operation.
        cast_input_types : bool, default: True
            If specified, the columns of the caller/callee query compilers will be assigned
            dtypes in the following priority, depending on what values were specified:
            (1) the `dtype` argument,
            (2) the dtype of the `out` array,
            (3) the common parent dtype of `self` and `other`.
            If this flag is not specified, then the resulting dtype is left to be determined
            by the result of the modin operation.
        dtype : numpy type, optional
            The desired dtype of the output array.
        out : array, optional
            Existing array object to which to assign the computation's result.

        Returns
        -------
        tuple
            Returns a 4-tuple with the following elements:
            - 0: QueryCompiler object that is the LHS of the binary operation, with types converted
                 as needed.
            - 1: QueryCompiler object OR scalar that is the RHS of the binary operation, with types
                 converted as needed.
            - 2: The ndim of the result.
            - 3: kwargs to pass to the query compiler.
        """
    other = try_convert_from_interoperable_type(other)
    if cast_input_types:
        operand_dtype = self.dtype if not isinstance(other, array) else pandas.core.dtypes.cast.find_common_type([self.dtype, other.dtype])
        out_dtype = dtype if dtype is not None else out.dtype if out is not None else operand_dtype
        self._query_compiler = self._query_compiler.astype({col_name: out_dtype for col_name in self._query_compiler.columns})
    if is_scalar(other):
        return (self._query_compiler, other, self._ndim, {})
    elif cast_input_types:
        other._query_compiler = other._query_compiler.astype({col_name: out_dtype for col_name in other._query_compiler.columns})
    if not isinstance(other, array):
        raise TypeError(f"Unsupported operand type(s): '{type(self)}' and '{type(other)}'")
    broadcast = self._ndim != other._ndim
    if broadcast:
        caller, callee = (self, other) if self._ndim == 2 else (other, self)
        if callee.shape[0] != caller.shape[1]:
            raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other.shape}')
        return (caller._query_compiler, callee._query_compiler, caller._ndim, {'broadcast': broadcast, 'axis': 1})
    elif self.shape != other.shape:
        broadcast = True
        if self.shape[0] == other.shape[0]:
            matched_dimension = 0
        elif self.shape[1] == other.shape[1]:
            matched_dimension = 1
            broadcast = False
        else:
            raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other.shape}')
        if self.shape[matched_dimension ^ 1] == 1 or other.shape[matched_dimension ^ 1] == 1:
            return (self._query_compiler, other._query_compiler, self._ndim, {'broadcast': broadcast, 'axis': matched_dimension})
        else:
            raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other.shape}')
    else:
        return (self._query_compiler, other._query_compiler, self._ndim, {'broadcast': False})