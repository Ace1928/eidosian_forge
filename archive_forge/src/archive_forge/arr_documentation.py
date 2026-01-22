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
        