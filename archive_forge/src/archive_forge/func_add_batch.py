import collections
from typing import Any, Mapping
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.pandas_block import PandasBlockBuilder
from ray.data.block import Block, BlockAccessor, DataBatch
def add_batch(self, batch: DataBatch):
    """Add a user-facing data batch to the builder.

        This data batch will be converted to an internal block and then added to the
        underlying builder.
        """
    block = BlockAccessor.batch_to_block(batch)
    return self.add_block(block)