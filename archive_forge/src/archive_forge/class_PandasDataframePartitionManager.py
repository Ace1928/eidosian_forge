import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
import pandas
from pandas._libs.lib import no_default
from modin.config import (
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
class PandasDataframePartitionManager(ClassLogger, ABC, modin_layer='PARTITION-MANAGER'):
    """
    Base class for managing the dataframe data layout and operators across the distribution of partitions.

    Partition class is the class to use for storing each partition.
    Each partition must extend the `PandasDataframePartition` class.
    """
    _partition_class = None
    _column_partitions_class = None
    _row_partition_class = None
    _execution_wrapper = None

    @classmethod
    def materialize_futures(cls, input_list):
        """
        Materialize all futures in the input list.

        Parameters
        ----------
        input_list : list
            The list that has to be manipulated.

        Returns
        -------
        list
           A new list with materialized objects.
        """
        if input_list is None:
            return None
        filtered_list = []
        filtered_idx = []
        for idx, item in enumerate(input_list):
            if cls._execution_wrapper.is_future(item):
                filtered_idx.append(idx)
                filtered_list.append(item)
        filtered_list = cls._execution_wrapper.materialize(filtered_list)
        result = input_list.copy()
        for idx, item in zip(filtered_idx, filtered_list):
            result[idx] = item
        return result

    @classmethod
    def preprocess_func(cls, map_func):
        """
        Preprocess a function to be applied to `PandasDataframePartition` objects.

        Parameters
        ----------
        map_func : callable
            The function to be preprocessed.

        Returns
        -------
        callable
            The preprocessed version of the `map_func` provided.

        Notes
        -----
        Preprocessing does not require any specific format, only that the
        `PandasDataframePartition.apply` method will recognize it (for the subclass
        being used).

        If your `PandasDataframePartition` objects assume that a function provided
        is serialized or wrapped or in some other format, this is the place
        to add that logic. It is possible that this can also just return
        `map_func` if the `apply` method of the `PandasDataframePartition` object
        you are using does not require any modification to a given function.
        """
        if cls._execution_wrapper.is_future(map_func):
            return map_func
        old_value = PersistentPickle.get()
        need_update = not PersistentPickle.get() and Engine.get() != 'Dask'
        if need_update:
            PersistentPickle.put(True)
        try:
            result = cls._partition_class.preprocess_func(map_func)
        finally:
            if need_update:
                PersistentPickle.put(old_value)
        return result

    @classmethod
    def create_partition_from_metadata(cls, **metadata):
        """
        Create NumPy array of partitions that holds an empty dataframe with given metadata.

        Parameters
        ----------
        **metadata : dict
            Metadata that has to be wrapped in a partition.

        Returns
        -------
        np.ndarray
            A NumPy 2D array of a single partition which contains the data.
        """
        metadata_dataframe = pandas.DataFrame(**metadata)
        return np.array([[cls._partition_class.put(metadata_dataframe)]])

    @classmethod
    def column_partitions(cls, partitions, full_axis=True):
        """
        Get the list of `BaseDataframeAxisPartition` objects representing column-wise partitions.

        Parameters
        ----------
        partitions : list-like
            List of (smaller) partitions to be combined to column-wise partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.

        Notes
        -----
        Each value in this list will be an `BaseDataframeAxisPartition` object.
        `BaseDataframeAxisPartition` is located in `axis_partition.py`.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [cls._column_partitions_class(col, full_axis=full_axis) for frame in partitions for col in frame.T]

    @classmethod
    def row_partitions(cls, partitions):
        """
        List of `BaseDataframeAxisPartition` objects representing row-wise partitions.

        Parameters
        ----------
        partitions : list-like
            List of (smaller) partitions to be combined to row-wise partitions.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.

        Notes
        -----
        Each value in this list will an `BaseDataframeAxisPartition` object.
        `BaseDataframeAxisPartition` is located in `axis_partition.py`.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [cls._row_partition_class(row) for frame in partitions for row in frame]

    @classmethod
    def axis_partition(cls, partitions, axis, full_axis: bool=True):
        """
        Logically partition along given axis (columns or rows).

        Parameters
        ----------
        partitions : list-like
            List of partitions to be combined.
        axis : {0, 1}
            0 for column partitions, 1 for row partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.
        """
        make_column_partitions = axis == 0
        if not full_axis and (not make_column_partitions):
            raise NotImplementedError("Row partitions must contain the entire axis. We don't " + 'support virtual partitioning for row partitions yet.')
        return cls.column_partitions(partitions) if make_column_partitions else cls.row_partitions(partitions)

    @classmethod
    def groupby_reduce(cls, axis, partitions, by, map_func, reduce_func, apply_indices=None):
        """
        Groupby data using the `map_func` provided along the `axis` over the `partitions` then reduce using `reduce_func`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to groupby over.
        partitions : NumPy 2D array
            Partitions of the ModinFrame to groupby.
        by : NumPy 2D array
            Partitions of 'by' to broadcast.
        map_func : callable
            Map function.
        reduce_func : callable,
            Reduce function.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.

        Returns
        -------
        NumPy array
            Partitions with applied groupby.
        """
        if apply_indices is not None:
            partitions = partitions[apply_indices] if axis else partitions[:, apply_indices]
        if by is not None:
            assert partitions.shape[axis] == by.shape[axis], f'the number of partitions along axis={axis!r} is not equal: ' + f'{partitions.shape[axis]} != {by.shape[axis]}'
            mapped_partitions = cls.broadcast_apply(axis, map_func, left=partitions, right=by)
        else:
            mapped_partitions = cls.map_partitions(partitions, map_func)
        num_splits = min(len(partitions), NPartitions.get())
        return cls.map_axis_partitions(axis, mapped_partitions, reduce_func, enumerate_partitions=True, num_splits=num_splits)

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_apply_select_indices(cls, axis, apply_func, left, right, left_indices, right_indices, keep_remaining=False):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` to selected indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : NumPy 2D array
            Left partitions.
        right : NumPy 2D array
            Right partitions.
        left_indices : list-like
            Indices to apply function to.
        right_indices : dictionary of indices of right partitions
            Indices that you want to bring at specified left partition, for example
            dict {key: {key1: [0, 1], key2: [5]}} means that in left[key] you want to
            broadcast [right[key1], right[key2]] partitions and internal indices
            for `right` must be [[0, 1], [5]].
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
        NumPy array
            An array of partition objects.

        Notes
        -----
        Your internal function must take these kwargs:
        [`internal_indices`, `other`, `internal_other_indices`] to work correctly!
        """
        if not axis:
            partitions_for_apply = left.T
            right = right.T
        else:
            partitions_for_apply = left
        [obj.drain_call_queue() for row in right for obj in row]

        def get_partitions(index):
            """Grab required partitions and indices from `right` and `right_indices`."""
            must_grab = right_indices[index]
            partitions_list = np.array([right[i] for i in must_grab.keys()])
            indices_list = list(must_grab.values())
            return {'other': partitions_list, 'internal_other_indices': indices_list}
        new_partitions = np.array([partitions_for_apply[i] if i not in left_indices else cls._apply_func_to_list_of_partitions_broadcast(apply_func, partitions_for_apply[i], internal_indices=left_indices[i], **get_partitions(i)) for i in range(len(partitions_for_apply)) if i in left_indices or keep_remaining])
        if not axis:
            new_partitions = new_partitions.T
        return new_partitions

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_apply(cls, axis, apply_func, left, right):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` function.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : np.ndarray
            NumPy array of left partitions.
        right : np.ndarray
            NumPy array of right partitions.

        Returns
        -------
        np.ndarray
            NumPy array of result partition objects.

        Notes
        -----
        This will often be overridden by implementations. It materializes the
        entire partitions of the right and applies them to the left through `apply`.
        """

        def map_func(df, *others):
            other = pandas.concat(others, axis=axis ^ 1) if len(others) > 1 else others[0]
            del others
            return apply_func(df, other)
        map_func = cls.preprocess_func(map_func)
        rt_axis_parts = cls.axis_partition(right, axis ^ 1)
        return np.array([[part.apply(map_func, *(rt_axis_parts[col_idx].list_of_blocks if axis else rt_axis_parts[row_idx].list_of_blocks)) for col_idx, part in enumerate(left[row_idx])] for row_idx in range(len(left))])

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_axis_partitions(cls, axis, apply_func, left, right, keep_partitioning=False, num_splits=None, apply_indices=None, enumerate_partitions=False, lengths=None, apply_func_args=None, **kwargs):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` along full `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : NumPy 2D array
            Left partitions.
        right : NumPy 2D array
            Right partitions.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `apply_func`.
            Note that `apply_func` must be able to accept `partition_idx` kwarg.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        apply_func_args : list-like, optional
            Positional arguments to pass to the `func`.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of partition objects.
        """
        ErrorMessage.catch_bugs_and_request_email(failure_condition=keep_partitioning and lengths is not None, extra_log=f'`keep_partitioning` must be set to `False` when passing `lengths`. Got: keep_partitioning={keep_partitioning!r} | lengths={lengths!r}')
        if keep_partitioning and num_splits is None:
            num_splits = len(left) if axis == 0 else len(left.T)
        elif lengths:
            num_splits = len(lengths)
        elif num_splits is None:
            num_splits = NPartitions.get()
        else:
            ErrorMessage.catch_bugs_and_request_email(failure_condition=not isinstance(num_splits, int), extra_log=f'Expected `num_splits` to be an integer, got: {type(num_splits)} | num_splits={num_splits!r}')
        preprocessed_map_func = cls.preprocess_func(apply_func)
        left_partitions = cls.axis_partition(left, axis)
        right_partitions = None if right is None else cls.axis_partition(right, axis)
        kw = {'num_splits': num_splits, 'other_axis_partition': right_partitions, 'maintain_partitioning': keep_partitioning}
        if lengths:
            kw['lengths'] = lengths
            kw['manual_partition'] = True
        if apply_indices is None:
            apply_indices = np.arange(len(left_partitions))
        result_blocks = np.array([left_partitions[i].apply(preprocessed_map_func, *(apply_func_args if apply_func_args else []), **kw, **{'partition_idx': idx} if enumerate_partitions else {}, **kwargs) for idx, i in enumerate(apply_indices)])
        return result_blocks.T if not axis else result_blocks

    @classmethod
    @wait_computations_if_benchmark_mode
    def map_partitions(cls, partitions, map_func, func_args=None, func_kwargs=None):
        """
        Apply `map_func` to every partition in `partitions`.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions housing the data of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array([[part.apply(preprocessed_map_func, *(func_args if func_args is not None else ()), **func_kwargs if func_kwargs is not None else {}) for part in row_of_parts] for row_of_parts in partitions])

    @classmethod
    @wait_computations_if_benchmark_mode
    def lazy_map_partitions(cls, partitions, map_func, func_args=None, func_kwargs=None, enumerate_partitions=False):
        """
        Apply `map_func` to every partition in `partitions` *lazily*.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.
        enumerate_partitions : bool, default: False

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array([[part.add_to_apply_calls(preprocessed_map_func, *(tuple() if func_args is None else func_args), **func_kwargs if func_kwargs is not None else {}, **{'partition_idx': i} if enumerate_partitions else {}) for part in row] for i, row in enumerate(partitions)])

    @classmethod
    def map_axis_partitions(cls, axis, partitions, map_func, keep_partitioning=False, num_splits=None, lengths=None, enumerate_partitions=False, **kwargs):
        """
        Apply `map_func` to every partition in `partitions` along given `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to perform the map across (0 - index, 1 - columns).
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `map_func`.
            Note that `map_func` must be able to accept `partition_idx` kwarg.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case when `map_func` relies on
        some global information about the axis.
        """
        return cls.broadcast_axis_partitions(axis=axis, left=partitions, apply_func=map_func, keep_partitioning=keep_partitioning, num_splits=num_splits, right=None, lengths=lengths, enumerate_partitions=enumerate_partitions, **kwargs)

    @classmethod
    def concat(cls, axis, left_parts, right_parts):
        """
        Concatenate the blocks of partitions with another set of blocks.

        Parameters
        ----------
        axis : int
            The axis to concatenate to.
        left_parts : np.ndarray
            NumPy array of partitions to concatenate with.
        right_parts : np.ndarray or list
            NumPy array of partitions to be concatenated.

        Returns
        -------
        np.ndarray
            A new NumPy array with concatenated partitions.
        list[int] or None
            Row lengths if possible to compute it.

        Notes
        -----
        Assumes that the blocks are already the same shape on the
        dimension being concatenated. A ValueError will be thrown if this
        condition is not met.
        """
        if type(right_parts) is list:
            right_parts = [o for o in right_parts if o.size != 0]
            to_concat = [left_parts] + right_parts if left_parts.size != 0 else right_parts
            result = np.concatenate(to_concat, axis=axis) if len(to_concat) else left_parts
        else:
            result = np.append(left_parts, right_parts, axis=axis)
        if axis == 0:
            return cls.rebalance_partitions(result)
        else:
            return (result, None)

    @classmethod
    def to_pandas(cls, partitions):
        """
        Convert NumPy array of PandasDataframePartition to pandas DataFrame.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array of PandasDataframePartition.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame
        """
        return create_pandas_df_from_partitions(cls.get_objects_from_partitions(partitions.flatten()), partitions.shape)

    @classmethod
    def to_numpy(cls, partitions, **kwargs):
        """
        Convert NumPy array of PandasDataframePartition to NumPy array of data stored within `partitions`.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array of PandasDataframePartition.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.to_numpy function.

        Returns
        -------
        np.ndarray
            A NumPy array.
        """
        return np.block([[block.to_numpy(**kwargs) for block in row] for row in partitions])

    @classmethod
    def split_pandas_df_into_partitions(cls, df, row_chunksize, col_chunksize, update_bar):
        """
        Split given pandas DataFrame according to the row/column chunk sizes into distributed partitions.

        Parameters
        ----------
        df : pandas.DataFrame
        row_chunksize : int
        col_chunksize : int
        update_bar : callable(x) -> x
            Function that updates a progress bar.

        Returns
        -------
        2D np.ndarray[PandasDataframePartition]
        """
        put_func = cls._partition_class.put
        if col_chunksize >= len(df.columns):
            col_parts = [df]
        else:
            col_parts = [df.iloc[:, i:i + col_chunksize] for i in range(0, len(df.columns), col_chunksize)]
        parts = [[update_bar(put_func(col_part.iloc[i:i + row_chunksize])) for col_part in col_parts] for i in range(0, len(df), row_chunksize)]
        return np.array(parts)

    @classmethod
    @wait_computations_if_benchmark_mode
    def from_pandas(cls, df, return_dims=False):
        """
        Return the partitions from pandas.DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas.DataFrame.
        return_dims : bool, default: False
            If it's True, return as (np.ndarray, row_lengths, col_widths),
            else np.ndarray.

        Returns
        -------
        np.ndarray or (np.ndarray, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """
        num_splits = NPartitions.get()
        min_block_size = MinPartitionSize.get()
        row_chunksize = compute_chunksize(df.shape[0], num_splits, min_block_size)
        col_chunksize = compute_chunksize(df.shape[1], num_splits, min_block_size)
        bar_format = '{l_bar}{bar}{r_bar}' if os.environ.get('DEBUG_PROGRESS_BAR', 'False') == 'True' else '{desc}: {percentage:3.0f}%{bar} Elapsed time: {elapsed}, estimated remaining time: {remaining}'
        if ProgressBar.get():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    from tqdm.autonotebook import tqdm as tqdm_notebook
                except ImportError:
                    raise ImportError('Please pip install tqdm to use the progress bar')
            rows = max(1, round(len(df) / row_chunksize))
            cols = max(1, round(len(df.columns) / col_chunksize))
            update_count = rows * cols
            pbar = tqdm_notebook(total=round(update_count), desc='Distributing Dataframe', bar_format=bar_format)
        else:
            pbar = None

        def update_bar(f):
            if ProgressBar.get():
                pbar.update(1)
            return f
        parts = cls.split_pandas_df_into_partitions(df, row_chunksize, col_chunksize, update_bar)
        if ProgressBar.get():
            pbar.close()
        if not return_dims:
            return parts
        else:
            row_lengths = [row_chunksize if i + row_chunksize < len(df) else len(df) % row_chunksize or row_chunksize for i in range(0, len(df), row_chunksize)]
            col_widths = [col_chunksize if i + col_chunksize < len(df.columns) else len(df.columns) % col_chunksize or col_chunksize for i in range(0, len(df.columns), col_chunksize)]
            return (parts, row_lengths, col_widths)

    @classmethod
    def from_arrow(cls, at, return_dims=False):
        """
        Return the partitions from Apache Arrow (PyArrow).

        Parameters
        ----------
        at : pyarrow.table
            Arrow Table.
        return_dims : bool, default: False
            If it's True, return as (np.ndarray, row_lengths, col_widths),
            else np.ndarray.

        Returns
        -------
        np.ndarray or (np.ndarray, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """
        return cls.from_pandas(at.to_pandas(), return_dims=return_dims)

    @classmethod
    def get_objects_from_partitions(cls, partitions):
        """
        Get the objects wrapped by `partitions` (in parallel if supported).

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
        if hasattr(cls, '_execution_wrapper'):
            for idx, part in enumerate(partitions):
                if hasattr(part, 'force_materialization'):
                    partitions[idx] = part.force_materialization()
            assert all([len(partition.list_of_blocks) == 1 for partition in partitions]), 'Implementation assumes that each partition contains a single block.'
            return cls._execution_wrapper.materialize([partition.list_of_blocks[0] for partition in partitions])
        return [partition.get() for partition in partitions]

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions`, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Notes
        -----
        This method should be implemented in a more efficient way for engines that supports
        waiting on objects in parallel.
        """
        for partition in partitions:
            partition.wait()

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        Get the internal indices stored in the partitions.

        Parameters
        ----------
        axis : {0, 1}
            Axis to extract the labels over.
        partitions : np.ndarray
            NumPy array with PandasDataframePartition's.
        index_func : callable, default: None
            The function to be used to extract the indices.

        Returns
        -------
        pandas.Index
            A pandas Index object.
        list of pandas.Index
            The list of internal indices for each partition.

        Notes
        -----
        These are the global indices of the object. This is mostly useful
        when you have deleted rows/columns internally, but do not know
        which ones were deleted.
        """
        if index_func is None:
            index_func = lambda df: df.axes[axis]
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        target = partitions.T if axis == 0 else partitions
        if len(target):
            new_idx = [idx.apply(func) for idx in target[0]]
            new_idx = cls.get_objects_from_partitions(new_idx)
        else:
            new_idx = [pandas.Index([])]
        total_idx = list(filter(len, new_idx))
        if len(total_idx) > 0:
            total_idx = total_idx[0].append(total_idx[1:])
        else:
            total_idx = new_idx[0]
        return (total_idx, new_idx)

    @classmethod
    def _apply_func_to_list_of_partitions_broadcast(cls, func, partitions, other, **kwargs):
        """
        Apply a function to a list of remote partitions.

        `other` partitions will be broadcasted to `partitions`
        and `func` will be applied.

        Parameters
        ----------
        func : callable
            The func to apply.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        other : np.ndarray
            The partitions to be broadcasted to `partitions`.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.apply function.

        Returns
        -------
        list
            A list of PandasDataframePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, other=[o.get() for o in broadcasted], **kwargs) for obj, broadcasted in zip(partitions, other.T)]

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """
        Apply a function to a list of remote partitions.

        Parameters
        ----------
        func : callable
            The func to apply.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.apply function.

        Returns
        -------
        list
            A list of PandasDataframePartition objects.

        Notes
        -----
        This preprocesses the `func` first before applying it to the partitions.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    @classmethod
    def combine(cls, partitions):
        """
        Convert a NumPy 2D array of partitions to a NumPy 2D array of a single partition.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions which have to be converted to a single partition.

        Returns
        -------
        np.ndarray
            A NumPy 2D array of a single partition.
        """
        if partitions.size <= 1:
            return partitions

        def to_pandas_remote(df, partition_shape, *dfs):
            """Copy of ``cls.to_pandas()`` method adapted for a remote function."""
            return create_pandas_df_from_partitions((df,) + dfs, partition_shape, called_from_remote=True)
        preprocessed_func = cls.preprocess_func(to_pandas_remote)
        partition_shape = partitions.shape
        partitions_flattened = partitions.flatten()
        for idx, part in enumerate(partitions_flattened):
            if hasattr(part, 'force_materialization'):
                partitions_flattened[idx] = part.force_materialization()
        partition_refs = [partition.list_of_blocks[0] for partition in partitions_flattened[1:]]
        combined_partition = partitions.flat[0].apply(preprocessed_func, partition_shape, *partition_refs)
        return np.array([combined_partition]).reshape(1, -1)

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_select_indices(cls, axis, partitions, func, indices, keep_remaining=False):
        """
        Apply a function to select indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply the `func` over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply to these indices of partitions.
        indices : dict
            The indices to apply the function to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions. Some operations
            may want to drop the remaining partitions and keep
            only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        Your internal function must take a kwarg `internal_indices` for
        this to work correctly. This prevents information leakage of the
        internal index to the external representation.
        """
        if partitions.size == 0:
            return np.array([[]])
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        if not axis:
            partitions_for_apply = partitions.T
        else:
            partitions_for_apply = partitions
        if dict_func is not None:
            if not keep_remaining:
                result = np.array([cls._apply_func_to_list_of_partitions(func, partitions_for_apply[o_idx], func_dict={i_idx: dict_func[i_idx] for i_idx in list_to_apply if i_idx >= 0}) for o_idx, list_to_apply in indices.items()])
            else:
                result = np.array([partitions_for_apply[i] if i not in indices else cls._apply_func_to_list_of_partitions(func, partitions_for_apply[i], func_dict={idx: dict_func[idx] for idx in indices[i] if idx >= 0}) for i in range(len(partitions_for_apply))])
        elif not keep_remaining:
            result = np.array([cls._apply_func_to_list_of_partitions(func, partitions_for_apply[idx], internal_indices=list_to_apply) for idx, list_to_apply in indices.items()])
        else:
            result = np.array([partitions_for_apply[i] if i not in indices else cls._apply_func_to_list_of_partitions(func, partitions_for_apply[i], internal_indices=indices[i]) for i in range(len(partitions_for_apply))])
        return result.T if not axis else result

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_select_indices_along_full_axis(cls, axis, partitions, func, indices, keep_remaining=False):
        """
        Apply a function to a select subset of full columns/rows.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        indices : list-like
            The global indices to apply the func to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        This should be used when you need to apply a function that relies
        on some global information for the entire column/row, but only need
        to apply a function to a subset.
        For your func to operate directly on the indices provided,
        it must use `internal_indices` as a keyword argument.
        """
        if partitions.size == 0:
            return np.array([[]])
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        preprocessed_func = cls.preprocess_func(func)
        if not keep_remaining:
            selected_partitions = partitions.T if not axis else partitions
            selected_partitions = np.array([selected_partitions[i] for i in indices])
            selected_partitions = selected_partitions.T if not axis else selected_partitions
        else:
            selected_partitions = partitions
        if not axis:
            partitions_for_apply = cls.column_partitions(selected_partitions)
            partitions_for_remaining = partitions.T
        else:
            partitions_for_apply = cls.row_partitions(selected_partitions)
            partitions_for_remaining = partitions
        if dict_func is not None:
            if not keep_remaining:
                result = np.array([part.apply(preprocessed_func, func_dict={idx: dict_func[idx] for idx in indices[i]}) for i, part in zip(indices, partitions_for_apply)])
            else:
                result = np.array([partitions_for_remaining[i] if i not in indices else cls._apply_func_to_list_of_partitions(preprocessed_func, partitions_for_apply[i], func_dict={idx: dict_func[idx] for idx in indices[i]}) for i in range(len(partitions_for_apply))])
        elif not keep_remaining:
            result = np.array([part.apply(preprocessed_func, internal_indices=indices[i]) for i, part in zip(indices, partitions_for_apply)])
        else:
            result = np.array([partitions_for_remaining[i] if i not in indices else partitions_for_apply[i].apply(preprocessed_func, internal_indices=indices[i]) for i in range(len(partitions_for_remaining))])
        return result.T if not axis else result

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_indices_both_axis(cls, partitions, func, row_partitions_list, col_partitions_list, item_to_distribute=no_default, row_lengths=None, col_widths=None):
        """
        Apply a function along both axes.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        row_partitions_list : iterable of tuples
            Iterable of tuples, containing 2 values:
                1. Integer row partition index.
                2. Internal row indexer of this partition.
        col_partitions_list : iterable of tuples
            Iterable of tuples, containing 2 values:
                1. Integer column partition index.
                2. Internal column indexer of this partition.
        item_to_distribute : np.ndarray or scalar, default: no_default
            The item to split up so it can be applied over both axes.
        row_lengths : list of ints, optional
            Lengths of partitions for every row. If not specified this information
            is extracted from partitions itself.
        col_widths : list of ints, optional
            Widths of partitions for every column. If not specified this information
            is extracted from partitions itself.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        For your func to operate directly on the indices provided,
        it must use `row_internal_indices`, `col_internal_indices` as keyword
        arguments.
        """
        partition_copy = partitions.copy()
        row_position_counter = 0
        if row_lengths is None:
            row_lengths = [None] * len(row_partitions_list)
        if col_widths is None:
            col_widths = [None] * len(col_partitions_list)

        def compute_part_size(indexer, remote_part, part_idx, axis):
            """Compute indexer length along the specified axis for the passed partition."""
            if isinstance(indexer, slice):
                shapes_container = row_lengths if axis == 0 else col_widths
                part_size = shapes_container[part_idx]
                if part_size is None:
                    part_size = remote_part.length() if axis == 0 else remote_part.width()
                    shapes_container[part_idx] = part_size
                indexer = range(*indexer.indices(part_size))
            return len(indexer)
        for row_idx, row_values in enumerate(row_partitions_list):
            row_blk_idx, row_internal_idx = row_values
            col_position_counter = 0
            row_offset = 0
            for col_idx, col_values in enumerate(col_partitions_list):
                col_blk_idx, col_internal_idx = col_values
                remote_part = partition_copy[row_blk_idx, col_blk_idx]
                row_offset = compute_part_size(row_internal_idx, remote_part, row_idx, axis=0)
                col_offset = compute_part_size(col_internal_idx, remote_part, col_idx, axis=1)
                if item_to_distribute is not no_default:
                    if isinstance(item_to_distribute, np.ndarray):
                        item = item_to_distribute[row_position_counter:row_position_counter + row_offset, col_position_counter:col_position_counter + col_offset]
                    else:
                        item = item_to_distribute
                    item = {'item': item}
                else:
                    item = {}
                block_result = remote_part.add_to_apply_calls(func, row_internal_indices=row_internal_idx, col_internal_indices=col_internal_idx, **item)
                partition_copy[row_blk_idx, col_blk_idx] = block_result
                col_position_counter += col_offset
            row_position_counter += row_offset
        return partition_copy

    @classmethod
    @wait_computations_if_benchmark_mode
    def n_ary_operation(cls, left, func, right: list):
        """
        Apply an n-ary operation to multiple ``PandasDataframe`` objects.

        This method assumes that all the partitions of the dataframes in left
        and right have the same dimensions. For each position i, j in each
        dataframe's partitions, the result has a partition at (i, j) whose data
        is func(left_partitions[i,j], \\*each_right_partitions[i,j]).

        Parameters
        ----------
        left : np.ndarray
            The partitions of left ``PandasDataframe``.
        func : callable
            The function to apply.
        right : list of np.ndarray
            The list of partitions of other ``PandasDataframe``.

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        func = cls.preprocess_func(func)

        def get_right_block(right_partitions, row_idx, col_idx):
            partition = right_partitions[row_idx][col_idx]
            blocks = partition.list_of_blocks
            "\n            NOTE:\n            Currently we do one remote call per right virtual partition to\n            materialize the partitions' blocks, then another remote call to do\n            the n_ary operation. we could get better performance if we\n            assembled the other partition within the remote `apply` call, by\n            passing the partition in as `other_axis_partition`. However,\n            passing `other_axis_partition` requires some extra care that would\n            complicate the code quite a bit:\n            - block partitions don't know how to deal with `other_axis_partition`\n            - the right axis partition's axis could be different from the axis\n              of the corresponding left partition\n            - there can be multiple other_axis_partition because this is an n-ary\n              operation and n can be > 2.\n            So for now just do the materialization in a separate remote step.\n            "
            if len(blocks) > 1:
                partition.force_materialization()
            assert len(partition.list_of_blocks) == 1
            return partition.list_of_blocks[0]
        return np.array([[part.apply(func, *(get_right_block(right_partitions, row_idx, col_idx) for right_partitions in right)) for col_idx, part in enumerate(left[row_idx])] for row_idx in range(len(left))])

    @classmethod
    def finalize(cls, partitions):
        """
        Perform all deferred calls on partitions.

        Parameters
        ----------
        partitions : np.ndarray
            Partitions of Modin Dataframe on which all deferred calls should be performed.
        """
        [part.drain_call_queue() for row in partitions for part in row]

    @classmethod
    def rebalance_partitions(cls, partitions):
        """
        Rebalance a 2-d array of partitions if we are using ``PandasOnRay`` or ``PandasOnDask`` executions.

        For all other executions, the partitions are returned unchanged.

        Rebalance the partitions by building a new array
        of partitions out of the original ones so that:

        - If all partitions have a length, each new partition has roughly the same number of rows.
        - Otherwise, each new partition spans roughly the same number of old partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to rebalance.

        Returns
        -------
        np.ndarray
            A NumPy array with the same; or new, rebalanced, partitions, depending on the execution
            engine and storage format.
        list[int] or None
            Row lengths if possible to compute it.
        """
        max_excess_of_num_partitions = 1.5
        num_existing_partitions = partitions.shape[0]
        ideal_num_new_partitions = NPartitions.get()
        if num_existing_partitions <= ideal_num_new_partitions * max_excess_of_num_partitions:
            return (partitions, None)
        if any((partition._length_cache is None for row in partitions for partition in row)):
            chunk_size = compute_chunksize(num_existing_partitions, ideal_num_new_partitions, min_block_size=1)
            new_partitions = np.array([cls.column_partitions(partitions[i:i + chunk_size], full_axis=False) for i in range(0, num_existing_partitions, chunk_size)])
            return (new_partitions, None)
        new_partitions = []
        start = 0
        total_rows = sum((part.length() for part in partitions[:, 0]))
        ideal_partition_size = compute_chunksize(total_rows, ideal_num_new_partitions, min_block_size=1)
        for _ in range(ideal_num_new_partitions):
            if start >= len(partitions):
                break
            stop = start
            partition_size = partitions[start][0].length()
            while stop < len(partitions) and partition_size < ideal_partition_size:
                stop += 1
                if stop < len(partitions):
                    partition_size += partitions[stop][0].length()
            if partition_size > ideal_partition_size * max_excess_of_num_partitions:
                prev_length = sum((row[0].length() for row in partitions[start:stop]))
                new_last_partition_size = ideal_partition_size - prev_length
                partitions = np.insert(partitions, stop + 1, [obj.mask(slice(new_last_partition_size, None), slice(None)) for obj in partitions[stop]], 0)
                for obj in partitions[stop + 1]:
                    obj._length_cache = partition_size - (prev_length + new_last_partition_size)
                partitions[stop, :] = [obj.mask(slice(None, new_last_partition_size), slice(None)) for obj in partitions[stop]]
                for obj in partitions[stop]:
                    obj._length_cache = new_last_partition_size
            new_partitions.append(cls.column_partitions(partitions[start:stop + 1], full_axis=False))
            start = stop + 1
        new_partitions = np.array(new_partitions)
        lengths = [part.length() for part in new_partitions[:, 0]]
        return (new_partitions, lengths)

    @classmethod
    @wait_computations_if_benchmark_mode
    def shuffle_partitions(cls, partitions, index, shuffle_functions: 'ShuffleFunctions', final_shuffle_func, right_partitions=None):
        """
        Return shuffled partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to shuffle.
        index : int or list of ints
            The index(es) of the column partitions corresponding to the partitions that contain the column to sample.
        shuffle_functions : ShuffleFunctions
            An object implementing the functions that we will be using to perform this shuffle.
        final_shuffle_func : Callable(pandas.DataFrame) -> pandas.DataFrame
            Function that shuffles the data within each new partition.
        right_partitions : np.ndarray, optional
            Partitions to broadcast to `self` partitions. If specified, the method builds range-partitioning
            for `right_partitions` basing on bins calculated for `partitions`, then performs broadcasting.

        Returns
        -------
        np.ndarray
            A list of row-partitions that have been shuffled.
        """
        masked_partitions = partitions[:, index]
        sample_func = cls.preprocess_func(shuffle_functions.sample_fn)
        if masked_partitions.ndim == 1:
            samples = [partition.apply(sample_func) for partition in masked_partitions]
        else:
            samples = [cls._row_partition_class(row_part, full_axis=False).apply(sample_func) for row_part in masked_partitions]
        samples = cls.get_objects_from_partitions(samples)
        num_bins = shuffle_functions.pivot_fn(samples)
        row_partitions = cls.row_partitions(partitions)
        if num_bins > 1:
            split_row_partitions = np.array([partition.split(shuffle_functions.split_fn, num_splits=num_bins, extract_metadata=False) for partition in row_partitions]).T
            if right_partitions is None:
                return np.array([[cls._column_partitions_class(row_partition, full_axis=False).apply(final_shuffle_func)] for row_partition in split_row_partitions])
            right_row_parts = cls.row_partitions(right_partitions)
            right_split_row_partitions = np.array([partition.split(shuffle_functions.split_fn, num_splits=num_bins, extract_metadata=False) for partition in right_row_parts]).T
            return np.array([cls._column_partitions_class(row_partition, full_axis=False).apply(final_shuffle_func, other_axis_partition=cls._column_partitions_class(right_row_partitions)) for right_row_partitions, row_partition in zip(right_split_row_partitions, split_row_partitions)])
        else:
            if right_partitions is None:
                return np.array([row_part.apply(final_shuffle_func) for row_part in row_partitions])
            right_row_parts = cls.row_partitions(right_partitions)
            return np.array([row_part.apply(final_shuffle_func, other_axis_partition=right_row_part) for right_row_part, row_part in zip(right_row_parts, row_partitions)])