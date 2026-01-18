import pandas
from distributed import Future
from distributed.utils import get_ip
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnDaskDataframePartition
Wait completing computations on the object wrapped by the partition.