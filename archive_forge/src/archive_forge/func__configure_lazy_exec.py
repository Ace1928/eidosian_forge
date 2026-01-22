from typing import TYPE_CHECKING, Callable, Union
import pandas
import ray
from modin.config import LazyExecution
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings
def _configure_lazy_exec(cls: LazyExecution):
    """Configure lazy execution mode for PandasOnRayDataframePartition."""
    mode = cls.get()
    get_logger().debug(f'Ray lazy execution mode: {mode}')
    if mode == 'Auto':
        PandasOnRayDataframePartition.apply = PandasOnRayDataframePartition._eager_exec_func
        PandasOnRayDataframePartition.add_to_apply_calls = PandasOnRayDataframePartition._lazy_exec_func
    elif mode == 'On':

        def lazy_exec(self, func, *args, **kwargs):
            return self._lazy_exec_func(func, *args, length=None, width=None, **kwargs)
        PandasOnRayDataframePartition.apply = lazy_exec
        PandasOnRayDataframePartition.add_to_apply_calls = PandasOnRayDataframePartition._lazy_exec_func
    elif mode == 'Off':

        def eager_exec(self, func, *args, length=None, width=None, **kwargs):
            return self._eager_exec_func(func, *args, **kwargs)
        PandasOnRayDataframePartition.apply = PandasOnRayDataframePartition._eager_exec_func
        PandasOnRayDataframePartition.add_to_apply_calls = eager_exec
    else:
        raise ValueError(f'Invalid lazy execution mode: {mode}')