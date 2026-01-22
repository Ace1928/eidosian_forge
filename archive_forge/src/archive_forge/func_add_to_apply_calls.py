import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
def add_to_apply_calls(self, func, *args, length=None, width=None, **kwargs):
    """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable or a future type
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        length : A future type or int, optional
            Length, or reference to it, of wrapped ``pandas.DataFrame``.
        width : A future type or int, optional
            Width, or reference to it, of wrapped ``pandas.DataFrame``.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasDataframeAxisPartition
            A new ``PandasDataframeAxisPartition`` object.
        """
    return type(self)(self.list_of_block_partitions, full_axis=self.full_axis, call_queue=self.call_queue + [[func, args, kwargs]], length=length, width=width)