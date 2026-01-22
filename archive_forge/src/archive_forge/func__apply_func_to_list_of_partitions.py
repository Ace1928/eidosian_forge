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