import pandas
import ray
from ray.util import get_node_ip_address
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
@_inherit_docstrings(PandasOnRayDataframeVirtualPartition.__init__)
class PandasOnRayDataframeRowPartition(PandasOnRayDataframeVirtualPartition):
    axis = 1