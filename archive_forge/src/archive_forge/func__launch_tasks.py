from typing import Tuple
import numpy as np
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
from modin.core.io import CSVDispatcher
@classmethod
def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
    """
        Launch tasks to read partitions.

        Parameters
        ----------
        splits : list
            List of tuples with partitions data, which defines
            parser task (start/end read bytes and etc).
        **partition_kwargs : dict
            Dictionary with keyword args that will be passed to the parser function.

        Returns
        -------
        partition_ids : list
            List with references to the partitions data.
        index_ids : list
            List with references to the partitions index objects.
        dtypes_ids : list
            List with references to the partitions dtypes objects.
        """
    partition_ids = [None] * len(splits)
    index_ids = [None] * len(splits)
    dtypes_ids = [None] * len(splits)
    gpu_manager = 0
    for idx, (start, end) in enumerate(splits):
        partition_kwargs.update({'start': start, 'end': end, 'gpu': gpu_manager})
        *partition_ids[idx], index_ids[idx], dtypes_ids[idx] = cls.deploy(func=cls.parse, f_kwargs=partition_kwargs, num_returns=partition_kwargs.get('num_splits') + 2)
        gpu_manager += 1
    return (partition_ids, index_ids, dtypes_ids)