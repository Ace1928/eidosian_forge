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