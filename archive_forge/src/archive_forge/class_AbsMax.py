import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
@PublicAPI
class AbsMax(_AggregateOnKeyBase):
    """Defines absolute max aggregation."""

    def __init__(self, on: Optional[str]=None, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        self._set_key_fn(on)
        on_fn = _to_on_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'abs_max({str(on)})'
        super().__init__(init=_null_wrap_init(lambda k: 0), merge=_null_wrap_merge(ignore_nulls, max), accumulate_row=_null_wrap_accumulate_row(ignore_nulls, on_fn, lambda a, r: max(a, abs(r))), finalize=_null_wrap_finalize(lambda a: a), name=self._rs_name)