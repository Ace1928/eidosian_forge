import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
class MapBatches(AbstractUDFMap):
    """Logical operator for map_batches."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, batch_size: Optional[int]=DEFAULT_BATCH_SIZE, batch_format: str='default', zero_copy_batch: bool=False, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, min_rows_per_block: Optional[int]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        super().__init__('MapBatches', input_op, fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs, min_rows_per_block=min_rows_per_block, compute=compute, ray_remote_args=ray_remote_args)
        self._batch_size = batch_size
        self._batch_format = batch_format
        self._zero_copy_batch = zero_copy_batch

    @property
    def can_modify_num_rows(self) -> bool:
        return False