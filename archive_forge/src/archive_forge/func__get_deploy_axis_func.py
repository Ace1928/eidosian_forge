import pandas
import ray
from ray.util import get_node_ip_address
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
@classmethod
def _get_deploy_axis_func(cls):
    if cls._DEPLOY_AXIS_FUNC is None:
        cls._DEPLOY_AXIS_FUNC = RayWrapper.put(PandasDataframeAxisPartition.deploy_axis_func)
    return cls._DEPLOY_AXIS_FUNC