import collections
from typing import Any, Mapping
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.pandas_block import PandasBlockBuilder
from ray.data.block import Block, BlockAccessor, DataBatch
def will_build_yield_copy(self) -> bool:
    if self._builder is None:
        return True
    return self._builder.will_build_yield_copy()