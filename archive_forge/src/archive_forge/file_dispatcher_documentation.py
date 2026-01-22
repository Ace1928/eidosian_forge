import os
import fsspec
import numpy as np
from pandas.io.common import is_fsspec_url, is_url
from modin.config import AsyncReadMode
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError

        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        row_lengths : list
            Partitions rows lengths.
        column_widths : list
            Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partition objects.
        