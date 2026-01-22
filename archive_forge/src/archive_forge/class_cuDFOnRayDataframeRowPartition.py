import cudf
from modin.core.execution.ray.common import RayWrapper
from .partition import cuDFOnRayDataframePartition
class cuDFOnRayDataframeRowPartition(cuDFOnRayDataframeAxisPartition):
    """
    The row partition implementation of ``cuDFOnRayDataframeAxisPartition``.

    Parameters
    ----------
    partitions : np.ndarray
        NumPy array with ``cuDFOnRayDataframePartition``-s.
    """
    axis = 1

    def reduce(self, func):
        """
        Reduce partitions along `self.axis` and apply `func`.

        Parameters
        ----------
        func : calalble
            A func to apply.

        Returns
        -------
        cuDFOnRayDataframePartition

        Notes
        -----
        Since we are using row partitions, we can bypass the Ray plasma
        store during axis reduce functions.
        """
        keys = [partition.get_key() for partition in self.partitions]
        gpu = self.partitions[0].get_gpu_manager()
        key = gpu.reduce_key_list.remote(keys, func)
        key = RayWrapper.materialize(key)
        return cuDFOnRayDataframePartition(gpu_manager=gpu, key=key)