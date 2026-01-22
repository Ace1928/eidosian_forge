import numpy as np
import ray
from modin.config import GpuCount, MinPartitionSize
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.generic.partitioning import (
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from .axis_partition import (
from .partition import cuDFOnRayDataframePartition
@classmethod
def _get_gpu_managers(cls):
    """
        Get list of gpu managers.

        Returns
        -------
        list
        """
    return GPU_MANAGERS