import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
@PublicAPI
class Min(_AggregateOnKeyBase):
    """Defines min aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'min({str(on)})'
        null_merge = _null_wrap_merge(ignore_nulls, min)
        super().__init__(init=_null_wrap_init(lambda k: float('inf')), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, lambda block: BlockAccessor.for_block(block).min(on, ignore_nulls), null_merge), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)