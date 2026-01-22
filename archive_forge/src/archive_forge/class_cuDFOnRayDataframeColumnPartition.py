import cudf
from modin.core.execution.ray.common import RayWrapper
from .partition import cuDFOnRayDataframePartition
class cuDFOnRayDataframeColumnPartition(cuDFOnRayDataframeAxisPartition):
    """
    The column partition implementation of ``cuDFOnRayDataframeAxisPartition``.

    Parameters
    ----------
    partitions : np.ndarray
        NumPy array with ``cuDFOnRayDataframePartition``-s.
    """
    axis = 0

    def reduce(self, func):
        """
        Reduce partitions along `self.axis` and apply `func`.

        Parameters
        ----------
        func : callable
            A func to apply.

        Returns
        -------
        cuDFOnRayDataframePartition
        """
        keys = [partition.get_key() for partition in self.partitions]
        gpu_managers = [partition.get_gpu_manager() for partition in self.partitions]
        head_gpu_manager = gpu_managers[0]
        cudf_dataframe_object_ids = [gpu_manager.get.remote(key) for gpu_manager, key in zip(gpu_managers, keys)]
        key = head_gpu_manager.reduce.remote(cudf_dataframe_object_ids, axis=self.axis, func=func)
        key = RayWrapper.materialize(key)
        result = cuDFOnRayDataframePartition(gpu_manager=head_gpu_manager, key=key)
        return result