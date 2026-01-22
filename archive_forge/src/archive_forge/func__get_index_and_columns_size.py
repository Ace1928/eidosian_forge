import warnings
import pandas
import unidist
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.common.utils import deserialize
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len
@unidist.remote(num_returns=2)
def _get_index_and_columns_size(df):
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