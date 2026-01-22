from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
class CorrCovBuilder:
    """Responsible for building pandas query compiler's methods computing correlation and covariance matrices."""

    class Method(Enum):
        """Enum specifying what method to use (either CORR for correlation or COV for covariance)."""
        CORR = 1
        COV = 2

    @classmethod
    def build_corr_method(cls) -> Callable[['PandasQueryCompiler', str, int, bool], 'PandasQueryCompiler']:
        """
        Build a query compiler method computing the correlation matrix.

        Returns
        -------
        callable(qc: PandasQueryCompiler, method: str, min_periods: int, numeric_only: bool) -> PandasQueryCompiler
            A callable matching the ``BaseQueryCompiler.corr`` signature and computing the correlation matrix.
        """

        def corr_method(qc: 'PandasQueryCompiler', method: str, min_periods: int=1, numeric_only: bool=True) -> 'PandasQueryCompiler':
            if method != 'pearson':
                return super(type(qc), qc).corr(method=method, min_periods=min_periods, numeric_only=numeric_only)
            if not numeric_only and qc._modin_frame.has_materialized_columns:
                new_index, new_columns = (qc._modin_frame.copy_columns_cache(), qc._modin_frame.copy_columns_cache())
                new_dtypes = pandas.Series(np.repeat(pandas.api.types.pandas_dtype('float'), len(new_columns)), index=new_columns)
            elif numeric_only and qc._modin_frame.has_materialized_dtypes:
                old_dtypes = qc._modin_frame.dtypes
                new_columns = old_dtypes[old_dtypes.map(is_numeric_dtype)].index
                new_index = new_columns.copy()
                new_dtypes = pandas.Series(np.repeat(pandas.api.types.pandas_dtype('float'), len(new_columns)), index=new_columns)
            else:
                new_index, new_columns, new_dtypes = (None, None, None)
            map, reduce = cls._build_map_reduce_methods(min_periods, method=cls.Method.CORR, numeric_only=numeric_only)
            reduced = qc._modin_frame.apply_full_axis(axis=1, func=map)
            result = reduced.combine_and_apply(func=reduce, new_index=new_index, new_columns=new_columns, new_dtypes=new_dtypes)
            return qc.__constructor__(result)
        return corr_method

    @classmethod
    def build_cov_method(cls) -> Callable[['PandasQueryCompiler', int, int], 'PandasQueryCompiler']:
        """
        Build a query compiler method computing the covariance matrix.

        Returns
        -------
        callable(qc: PandasQueryCompiler, min_periods: int, ddof: int) -> PandasQueryCompiler
            A callable matching the ``BaseQueryCompiler.cov`` signature and computing the covariance matrix.
        """
        raise NotImplementedError('Computing covariance is not yet implemented.')

    @classmethod
    def _build_map_reduce_methods(cls, min_periods: int, method: Method, numeric_only: bool) -> Tuple[Callable[[pandas.DataFrame], pandas.DataFrame], Callable[[pandas.DataFrame], pandas.DataFrame]]:
        """
        Build MapReduce kernels for the specified corr/cov method.

        Parameters
        ----------
        min_periods : int
            The parameter to pass to the reduce method.
        method : CorrCovBuilder.Method
            Whether the kernels compute correlation or covariance.
        numeric_only : bool
            Whether to only include numeric types.

        Returns
        -------
        Tuple[Callable(pandas.DataFrame) -> pandas.DataFrame, Callable(pandas.DataFrame) -> pandas.DataFrame]
            A tuple holding the Map (at the first position) and the Reduce (at the second position) kernels
            computing correlation/covariance matrix.
        """
        if method == cls.Method.COV:
            raise NotImplementedError('Computing covariance is not yet implemented.')
        return (lambda df: _CorrCovKernels.map(df, numeric_only), lambda df: _CorrCovKernels.reduce(df, min_periods, method))