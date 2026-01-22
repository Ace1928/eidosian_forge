import numpy as np
from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.core.execution.ray.common import RayWrapper
class GenericRayDataframePartitionManager(PandasDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    @classmethod
    def to_numpy(cls, partitions, **kwargs):
        """
        Convert `partitions` into a NumPy array.

        Parameters
        ----------
        partitions : NumPy array
            A 2-D array of partitions to convert to local NumPy array.
        **kwargs : dict
            Keyword arguments to pass to each partition ``.to_numpy()`` call.

        Returns
        -------
        NumPy array
        """
        if partitions.shape[1] == 1:
            parts = cls.get_objects_from_partitions(partitions.flatten())
            parts = [part.to_numpy(**kwargs) for part in parts]
        else:
            parts = RayWrapper.materialize([obj.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).list_of_blocks[0] for row in partitions for obj in row])
        rows, cols = partitions.shape
        parts = [parts[i * cols:(i + 1) * cols] for i in range(rows)]
        return np.block(parts)