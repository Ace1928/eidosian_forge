from modin.core.execution.modin_aqp import progress_bar_wrapper
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.generic.partitioning import (
from .partition import PandasOnUnidistDataframePartition
from .virtual_partition import (
class PandasOnUnidistDataframePartitionManager(GenericUnidistDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""
    _partition_class = PandasOnUnidistDataframePartition
    _column_partitions_class = PandasOnUnidistDataframeColumnPartition
    _row_partition_class = PandasOnUnidistDataframeRowPartition
    _execution_wrapper = UnidistWrapper

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions` in parallel, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.
        """
        UnidistWrapper.wait([block for partition in partitions for block in partition.list_of_blocks])