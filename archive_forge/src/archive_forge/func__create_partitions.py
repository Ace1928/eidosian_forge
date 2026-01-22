import numpy as np
import ray
from modin.config import GpuCount, MinPartitionSize
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.generic.partitioning import (
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from .axis_partition import (
from .partition import cuDFOnRayDataframePartition
@classmethod
def _create_partitions(cls, keys, gpu_managers):
    """
        Create NumPy array of partitions.

        Parameters
        ----------
        keys : list
            List of keys associated with dataframes in
            `gpu_managers`.
        gpu_managers : list
            List of ``GPUManager`` objects, which store
            dataframes.

        Returns
        -------
        np.ndarray
            A NumPy array of ``cuDFOnRayDataframePartition`` objects.
        """
    return np.array([cls._partition_class(gpu_managers[i], keys[i]) for i in range(len(gpu_managers))])