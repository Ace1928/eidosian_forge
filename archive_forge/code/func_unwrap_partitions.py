from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from pandas._typing import Axes
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.pandas.dataframe import DataFrame, Series
def unwrap_partitions(api_layer_object: Union[DataFrame, Series], axis: Optional[int]=None, get_ip: bool=False) -> list:
    """
    Unwrap partitions of the ``api_layer_object``.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    axis : {None, 0, 1}, default: None
        The axis to unwrap partitions for (0 - row partitions, 1 - column partitions).
        If ``axis is None``, the partitions are unwrapped as they are currently stored.
    get_ip : bool, default: False
        Whether to get node ip address to each partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef/Dask.Future to partitions of the ``api_layer_object``
        if Ray/Dask is used as an engine.

    Notes
    -----
    If ``get_ip=True``, a list of tuples of Ray.ObjectRef/Dask.Future to node ip addresses and
    partitions of the ``api_layer_object``, respectively, is returned if Ray/Dask is used as an engine
    (i.e. ``[(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]``).
    """
    if not hasattr(api_layer_object, '_query_compiler'):
        raise ValueError(f'Only API Layer objects may be passed in here, got {type(api_layer_object)} instead.')
    modin_frame = api_layer_object._query_compiler._modin_frame
    modin_frame._propagate_index_objs(None)
    if axis is None:

        def _unwrap_partitions() -> list:
            [p.drain_call_queue() for p in modin_frame._partitions.flatten()]

            def get_block(partition: PartitionUnionType) -> np.ndarray:
                if hasattr(partition, 'force_materialization'):
                    blocks = partition.force_materialization().list_of_blocks
                else:
                    blocks = partition.list_of_blocks
                assert len(blocks) == 1, f'Implementation assumes that partition contains a single block, but {len(blocks)} recieved.'
                return blocks[0]
            if get_ip:
                return [[(partition.ip(materialize=False), get_block(partition)) for partition in row] for row in modin_frame._partitions]
            else:
                return [[get_block(partition) for partition in row] for row in modin_frame._partitions]
        actual_engine = type(api_layer_object._query_compiler._modin_frame._partitions[0][0]).__name__
        if actual_engine in ('PandasOnRayDataframePartition', 'PandasOnDaskDataframePartition', 'PandasOnUnidistDataframePartition', 'PandasOnRayDataframeColumnPartition', 'PandasOnRayDataframeRowPartition', 'PandasOnDaskDataframeColumnPartition', 'PandasOnDaskDataframeRowPartition', 'PandasOnUnidistDataframeColumnPartition', 'PandasOnUnidistDataframeRowPartition'):
            return _unwrap_partitions()
        raise ValueError(f"Do not know how to unwrap '{actual_engine}' underlying partitions")
    else:
        partitions = modin_frame._partition_mgr_cls.axis_partition(modin_frame._partitions, axis ^ 1)
        return [part.force_materialization(get_ip=get_ip).unwrap(squeeze=True, get_ip=get_ip) for part in partitions]