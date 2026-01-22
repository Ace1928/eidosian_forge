import warnings
import pandas
import unidist
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.common.utils import deserialize
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len
@unidist.remote(num_returns=4)
def _apply_list_of_funcs(call_queue, partition):
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    call_queue : list
        A call queue that needs to be executed on the partition.
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    int
        The number of rows of the resulting pandas DataFrame.
    int
        The number of columns of the resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    for func, f_args, f_kwargs in call_queue:
        func = deserialize(func)
        args = deserialize(f_args)
        kwargs = deserialize(f_kwargs)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                partition = func(partition, *args, **kwargs)
        except ValueError:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                partition = func(partition.copy(), *args, **kwargs)
    return (partition, len(partition) if hasattr(partition, '__len__') else 0, len(partition.columns) if hasattr(partition, 'columns') else 0, unidist.get_ip())