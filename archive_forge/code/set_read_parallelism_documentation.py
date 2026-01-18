import math
from typing import Optional, Tuple, Union
from ray import available_resources as ray_available_resources
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _autodetect_parallelism
from ray.data.context import WARN_PREFIX, DataContext
from ray.data.datasource.datasource import Datasource, Reader

    This rule sets the read op's task parallelism based on the target block
    size, the requested parallelism, the number of read files, and the
    available resources in the cluster.

    If the parallelism is lower than requested, this rule also sets a split
    factor to split the output blocks of the read task, so that the following
    stage will have the desired parallelism.
    