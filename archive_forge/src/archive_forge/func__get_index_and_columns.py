from typing import TYPE_CHECKING, Callable, Union
import pandas
import ray
from modin.config import LazyExecution
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings
@ray.remote(num_returns=2)
def _get_index_and_columns(df):
    """
    Get the number of rows and columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame which dimensions are needed.

    Returns
    -------
    int
        The number of rows.
    int
        The number of columns.
    """
    return (len(df.index), len(df.columns))