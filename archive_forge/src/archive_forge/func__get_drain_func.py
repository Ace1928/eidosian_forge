import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
@classmethod
def _get_drain_func(cls):
    return PandasDataframeAxisPartition.drain