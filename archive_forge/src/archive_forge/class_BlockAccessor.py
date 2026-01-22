import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
@DeveloperAPI
class BlockAccessor:
    """Provides accessor methods for a specific block.

    Ideally, we wouldn't need a separate accessor classes for blocks. However,
    this is needed if we want to support storing ``pyarrow.Table`` directly
    as a top-level Ray object, without a wrapping class (issue #17186).
    """

    def num_rows(self) -> int:
        """Return the number of rows contained in this block."""
        raise NotImplementedError

    def iter_rows(self, public_row_format: bool) -> Iterator[T]:
        """Iterate over the rows of this block.

        Args:
            public_row_format: Whether to cast rows into the public Dict row
                format (this incurs extra copy conversions).
        """
        raise NotImplementedError

    def slice(self, start: int, end: int, copy: bool) -> Block:
        """Return a slice of this block.

        Args:
            start: The starting index of the slice.
            end: The ending index of the slice.
            copy: Whether to perform a data copy for the slice.

        Returns:
            The sliced block result.
        """
        raise NotImplementedError

    def take(self, indices: List[int]) -> Block:
        """Return a new block containing the provided row indices.

        Args:
            indices: The row indices to return.

        Returns:
            A new block containing the provided row indices.
        """
        raise NotImplementedError

    def select(self, columns: List[Optional[str]]) -> Block:
        """Return a new block containing the provided columns."""
        raise NotImplementedError

    def random_shuffle(self, random_seed: Optional[int]) -> Block:
        """Randomly shuffle this block."""
        raise NotImplementedError

    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert this block into a Pandas dataframe."""
        raise NotImplementedError

    def to_numpy(self, columns: Optional[Union[str, List[str]]]=None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Convert this block (or columns of block) into a NumPy ndarray.

        Args:
            columns: Name of columns to convert, or None if converting all columns.
        """
        raise NotImplementedError

    def to_arrow(self) -> 'pyarrow.Table':
        """Convert this block into an Arrow table."""
        raise NotImplementedError

    def to_block(self) -> Block:
        """Return the base block that this accessor wraps."""
        raise NotImplementedError

    def to_default(self) -> Block:
        """Return the default data format for this accessor."""
        return self.to_block()

    def to_batch_format(self, batch_format: Optional[str]) -> DataBatch:
        """Convert this block into the provided batch format.

        Args:
            batch_format: The batch format to convert this block to.

        Returns:
            This block formatted as the provided batch format.
        """
        if batch_format is None:
            return self.to_block()
        elif batch_format == 'default' or batch_format == 'native':
            return self.to_default()
        elif batch_format == 'pandas':
            return self.to_pandas()
        elif batch_format == 'pyarrow':
            return self.to_arrow()
        elif batch_format == 'numpy':
            return self.to_numpy()
        else:
            raise ValueError(f'The batch format must be one of {VALID_BATCH_FORMATS}, got: {batch_format}')

    def size_bytes(self) -> int:
        """Return the approximate size in bytes of this block."""
        raise NotImplementedError

    def schema(self) -> Union[type, 'pyarrow.lib.Schema']:
        """Return the Python type or pyarrow schema of this block."""
        raise NotImplementedError

    def get_metadata(self, input_files: List[str], exec_stats: Optional[BlockExecStats]) -> BlockMetadata:
        """Create a metadata object from this block."""
        return BlockMetadata(num_rows=self.num_rows(), size_bytes=self.size_bytes(), schema=self.schema(), input_files=input_files, exec_stats=exec_stats)

    def zip(self, other: 'Block') -> 'Block':
        """Zip this block with another block of the same type and size."""
        raise NotImplementedError

    @staticmethod
    def builder() -> 'BlockBuilder':
        """Create a builder for this block type."""
        raise NotImplementedError

    @staticmethod
    def batch_to_block(batch: DataBatch) -> Block:
        """Create a block from user-facing data formats."""
        if isinstance(batch, np.ndarray):
            raise ValueError(f"Error validating {_truncated_repr(batch)}: Standalone numpy arrays are not allowed in Ray 2.5. Return a dict of field -> array, e.g., `{{'data': array}}` instead of `array`.")
        elif isinstance(batch, collections.abc.Mapping):
            import pyarrow as pa
            from ray.data._internal.arrow_block import ArrowBlockAccessor
            try:
                return ArrowBlockAccessor.numpy_to_block(batch)
            except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
                import pandas as pd
                return pd.DataFrame(dict(batch))
        return batch

    @staticmethod
    def for_block(block: Block) -> 'BlockAccessor[T]':
        """Create a block accessor for the given block."""
        _check_pyarrow_version()
        import pandas
        import pyarrow
        if isinstance(block, pyarrow.Table):
            from ray.data._internal.arrow_block import ArrowBlockAccessor
            return ArrowBlockAccessor(block)
        elif isinstance(block, pandas.DataFrame):
            from ray.data._internal.pandas_block import PandasBlockAccessor
            return PandasBlockAccessor(block)
        elif isinstance(block, bytes):
            from ray.data._internal.arrow_block import ArrowBlockAccessor
            return ArrowBlockAccessor.from_bytes(block)
        elif isinstance(block, list):
            raise ValueError(f"Error validating {_truncated_repr(block)}: Standalone Python objects are not allowed in Ray 2.5. To use Python objects in a dataset, wrap them in a dict of numpy arrays, e.g., return `{{'item': batch}}` instead of just `batch`.")
        else:
            raise TypeError('Not a block type: {} ({})'.format(block, type(block)))

    def sample(self, n_samples: int, sort_key: 'SortKey') -> 'Block':
        """Return a random sample of items from this block."""
        raise NotImplementedError

    def sort_and_partition(self, boundaries: List[T], sort_key: 'SortKey') -> List['Block']:
        """Return a list of sorted partitions of this block."""
        raise NotImplementedError

    def combine(self, key: Optional[str], agg: 'AggregateFn') -> Block:
        """Combine rows with the same key into an accumulator."""
        raise NotImplementedError

    @staticmethod
    def merge_sorted_blocks(blocks: List['Block'], sort_key: 'SortKey') -> Tuple[Block, BlockMetadata]:
        """Return a sorted block by merging a list of sorted blocks."""
        raise NotImplementedError

    @staticmethod
    def aggregate_combined_blocks(blocks: List[Block], key: Optional[str], agg: 'AggregateFn') -> Tuple[Block, BlockMetadata]:
        """Aggregate partially combined and sorted blocks."""
        raise NotImplementedError