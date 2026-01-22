from typing import Tuple
import numpy as np
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
from modin.core.io import CSVDispatcher

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
        