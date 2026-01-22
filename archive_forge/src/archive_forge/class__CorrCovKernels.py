from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
class _CorrCovKernels:
    """Holds kernel functions computing correlation/covariance matrices in a MapReduce manner."""

    @classmethod
    def map(cls, df: pandas.DataFrame, numeric_only: bool) -> pandas.DataFrame:
        """
        Perform the Map phase to compute the corr/cov matrix.

        In this kernel we compute all the required components to compute
        the correlation matrix at the reduce phase, the required components are:
            1. Matrix holding sums of pairwise multiplications between all columns
               defined as ``M[col1, col2] = sum(col1[i] * col2[i] for i in range(col_len))``
            2. Sum for each column (special case if there are NaN values)
            3. Sum of squares for each column (special case if there are NaN values)
            4. Number of values in each column (special case if there are NaN values)

        Parameters
        ----------
        df : pandas.DataFrame
            Partition to compute the aggregations for.
        numeric_only : bool
            Whether to only include numeric types.

        Returns
        -------
        pandas.DataFrame
            A MultiIndex columned DataFrame holding the described aggregation results for this
            specifix partition under the following keys: ``["mul", "sum", "pow2_sum", "count"]``
        """
        if numeric_only:
            df = df.select_dtypes(include='number')
        raw_df = df.values.T
        try:
            nan_mask = np.isnan(raw_df)
        except TypeError as e:
            raise ValueError("Unsupported types with 'numeric_only=False'") from e
        has_nans = nan_mask.sum() != 0
        if has_nans:
            if not raw_df.flags.writeable:
                raw_df = raw_df.copy()
            np.putmask(raw_df, nan_mask, values=0)
        cols = df.columns
        sum_of_pairwise_mul = pandas.DataFrame(np.dot(raw_df, raw_df.T), index=cols, columns=cols, copy=False)
        if has_nans:
            sums, sums_of_squares, count = cls._compute_nan_aggs(raw_df, cols, nan_mask)
        else:
            sums, sums_of_squares, count = cls._compute_non_nan_aggs(df)
        aggregations = pandas.concat([sum_of_pairwise_mul, sums, sums_of_squares, count], copy=False, axis=1, keys=['mul', 'sum', 'pow2_sum', 'count'])
        return aggregations

    @staticmethod
    def _compute_non_nan_aggs(df: pandas.DataFrame) -> Tuple[pandas.Series, pandas.Series, pandas.Series]:
        """
        Compute sums, sums of square and the number of observations for a partition assuming there are no NaN values in it.

        Parameters
        ----------
        df : pandas.DataFrame
            Partition to compute the aggregations for.

        Returns
        -------
        Tuple[sums: pandas.Series, sums_of_squares: pandas.Series, count: pandas.Series]
            A tuple storing Series where each of them holds the result for
            one of the described aggregations.
        """
        sums = df.sum().rename(MODIN_UNNAMED_SERIES_LABEL)
        sums_of_squares = (df ** 2).sum().rename(MODIN_UNNAMED_SERIES_LABEL)
        count = pandas.Series(np.repeat(len(df), len(df.columns)), index=df.columns, copy=False).rename(MODIN_UNNAMED_SERIES_LABEL)
        return (sums, sums_of_squares, count)

    @staticmethod
    def _compute_nan_aggs(raw_df: np.ndarray, cols: pandas.Index, nan_mask: np.ndarray) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        Compute sums, sums of square and the number of observations for a partition assuming there are NaN values in it.

        Parameters
        ----------
        raw_df : np.ndarray
            Raw values of the partition to compute the aggregations for.
        cols : pandas.Index
            Columns of the partition.
        nan_mask : np.ndarray[bool]
            Boolean mask showing positions of NaN values in the `raw_df`.

        Returns
        -------
        Tuple[sums: pandas.DataFrame, sums_of_squares: pandas.DataFrame, count: pandas.DataFrame]
            A tuple storing DataFrames where each of them holds the result for
            one of the described aggregations.
        """
        sums = {}
        sums_of_squares = {}
        count = {}
        for i, col in enumerate(cols):
            col_vals = np.resize(raw_df[i], raw_df.shape)
            np.putmask(col_vals, nan_mask, values=0)
            sums[col] = pandas.Series(np.sum(col_vals, axis=1), index=cols, copy=False)
            sums_of_squares[col] = pandas.Series(np.sum(col_vals ** 2, axis=1), index=cols, copy=False)
            count[col] = pandas.Series(nan_mask.shape[1] - np.count_nonzero(nan_mask | nan_mask[i], axis=1), index=cols, copy=False)
        sums = pandas.concat(sums, axis=1, copy=False)
        sums_of_squares = pandas.concat(sums_of_squares, axis=1, copy=False)
        count = pandas.concat(count, axis=1, copy=False)
        return (sums, sums_of_squares, count)

    @classmethod
    def reduce(cls, df: pandas.DataFrame, min_periods: int, method: CorrCovBuilder.Method) -> pandas.DataFrame:
        """
        Perform the Reduce phase to compute the corr/cov matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe holding aggregations computed for each partition
            concatenated along the rows axis.
        min_periods : int
            Minimum number of observations required per pair of columns to have a valid result.
        method : CorrCovBuilder.Method
            Whether to build a correlation or a covariance matrix.

        Returns
        -------
        pandas.DataFrame
            Either correlation or covariance matrix.
        """
        if method == CorrCovBuilder.Method.COV:
            raise NotImplementedError('Computing covariance is not yet implemented.')
        total_agg = df.groupby(level=0).sum()
        total_agg = cls._maybe_combine_nan_and_non_nan_aggs(total_agg)
        sum_of_pairwise_mul = total_agg['mul']
        sums = total_agg['sum']
        sums_of_squares = total_agg['pow2_sum']
        count = total_agg['count']
        cols = sum_of_pairwise_mul.columns
        has_nans = len(sums.columns) > 1
        if not has_nans:
            count = count.iloc[0, 0]
            if count < min_periods:
                return pandas.DataFrame(index=cols, columns=cols, dtype='float')
            sums = sums.squeeze(axis=1)
            sums_of_squares = sums_of_squares.squeeze(axis=1)
        means = sums / count
        std = np.sqrt(sums_of_squares - 2 * means * sums + count * means ** 2)
        if has_nans:
            return cls._build_corr_table_nan(sum_of_pairwise_mul, means, sums, count, std, cols, min_periods)
        else:
            return cls._build_corr_table_non_nan(sum_of_pairwise_mul, means, sums, count, std, cols)

    @staticmethod
    def _maybe_combine_nan_and_non_nan_aggs(total_agg: pandas.DataFrame) -> pandas.DataFrame:
        """
        Pair the aggregation results of partitions having and not having NaN values if needed.

        Parameters
        ----------
        total_agg : pandas.DataFrame
            A dataframe holding aggregations computed for each partition
            concatenated along the rows axis.

        Returns
        -------
        pandas.DataFrame
            DataFrame with aligned results.
        """
        nsums = total_agg.columns.get_locs(['sum'])
        if not (len(nsums) > 1 and ('sum', MODIN_UNNAMED_SERIES_LABEL) in total_agg.columns):
            return total_agg
        cols = total_agg.columns
        all_agg_idxs = np.where(cols.get_loc('sum') | cols.get_loc('pow2_sum') | cols.get_loc('count'))[0]
        non_na_agg_idxs = cols.get_indexer_for(pandas.Index([('sum', MODIN_UNNAMED_SERIES_LABEL), ('pow2_sum', MODIN_UNNAMED_SERIES_LABEL), ('count', MODIN_UNNAMED_SERIES_LABEL)]))
        na_agg_idxs = np.setdiff1d(all_agg_idxs, non_na_agg_idxs, assume_unique=True)
        parts_with_nans = total_agg.values[:, na_agg_idxs]
        parts_without_nans = total_agg.values[:, non_na_agg_idxs].repeat(repeats=len(parts_with_nans), axis=0).reshape(parts_with_nans.shape, order='F')
        replace_values = parts_with_nans + parts_without_nans
        if not total_agg.values.flags.writeable:
            total_agg = total_agg.copy()
        total_agg.values[:, na_agg_idxs] = replace_values
        return total_agg

    @staticmethod
    def _build_corr_table_nan(sum_of_pairwise_mul: pandas.DataFrame, means: pandas.DataFrame, sums: pandas.DataFrame, count: pandas.DataFrame, std: pandas.DataFrame, cols: pandas.Index, min_periods: int) -> pandas.DataFrame:
        """
        Build correlation matrix for a DataFrame that had NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.DataFrame
        sums : pandas.DataFrame
        count : pandas.DataFrame
        std : pandas.DataFrame
        cols : pandas.Index
        min_periods : int

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """
        res = pandas.DataFrame(index=cols, columns=cols, dtype='float')
        nan_mask = count < min_periods
        for col in cols:
            top = sum_of_pairwise_mul.loc[col] - sums.loc[col] * means[col] - means.loc[col] * sums[col] + count.loc[col] * means.loc[col] * means[col]
            down = std.loc[col] * std[col]
            res.loc[col, :] = top / down
        res[nan_mask] = np.nan
        return res

    @staticmethod
    def _build_corr_table_non_nan(sum_of_pairwise_mul: pandas.DataFrame, means: pandas.Series, sums: pandas.Series, count: int, std: pandas.Series, cols: pandas.Index) -> pandas.DataFrame:
        """
        Build correlation matrix for a DataFrame that didn't have NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.Series
        sums : pandas.Series
        count : int
        std : pandas.Series
        cols : pandas.Index

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """
        res = pandas.DataFrame(index=cols, columns=cols, dtype='float')
        for col in cols:
            top = sum_of_pairwise_mul.loc[col] - sums.loc[col] * means - means.loc[col] * sums + count * means.loc[col] * means
            down = std.loc[col] * std
            res.loc[col, :] = top / down
        return res