from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
class ShufflingBatcher(BatcherInterface):
    """Chunks blocks into shuffled batches, using a local in-memory shuffle buffer."""

    def __init__(self, batch_size: Optional[int], shuffle_buffer_min_size: int, shuffle_seed: Optional[int]=None):
        """Constructs a random-shuffling block batcher.

        Args:
            batch_size: Record batch size.
            shuffle_buffer_min_size: Minimum number of rows that must be in the local
                in-memory shuffle buffer in order to yield a batch. When there are no
                more rows to be added to the buffer, the number of rows in the buffer
                *will* decrease below this value while yielding the remaining batches,
                and the final batch may have less than ``batch_size`` rows. Increasing
                this will improve the randomness of the shuffle but may increase the
                latency to the first batch.
            shuffle_seed: The seed to use for the local random shuffle.
        """
        if batch_size is None:
            raise ValueError('Must specify a batch_size if using a local shuffle.')
        self._batch_size = batch_size
        self._shuffle_seed = shuffle_seed
        if shuffle_buffer_min_size < batch_size:
            shuffle_buffer_min_size = batch_size
        self._buffer_capacity = max(2 * shuffle_buffer_min_size, shuffle_buffer_min_size + batch_size)
        self._buffer_min_size = shuffle_buffer_min_size
        self._builder = DelegatingBlockBuilder()
        self._shuffle_buffer: Block = None
        self._batch_head = 0
        self._done_adding = False

    def add(self, block: Block):
        """Add a block to the shuffle buffer.

        Note empty block is not added to buffer.

        Args:
            block: Block to add to the shuffle buffer.
        """
        if BlockAccessor.for_block(block).num_rows() > 0:
            self._builder.add_block(block)

    def done_adding(self) -> bool:
        """Indicate to the batcher that no more blocks will be added to the batcher.

        No more blocks should be added to the batcher after calling this.
        """
        self._done_adding = True

    def has_any(self) -> bool:
        """Whether this batcher has any data."""
        return self._buffer_size() > 0

    def has_batch(self) -> bool:
        """Whether this batcher has any batches."""
        buffer_size = self._buffer_size()
        if not self._done_adding:
            return self._materialized_buffer_size() >= self._buffer_min_size or buffer_size - self._batch_size >= self._buffer_min_size * SHUFFLE_BUFFER_COMPACTION_RATIO
        else:
            return buffer_size >= self._batch_size

    def _buffer_size(self) -> int:
        """Return shuffle buffer size."""
        buffer_size = self._builder.num_rows()
        buffer_size += self._materialized_buffer_size()
        return buffer_size

    def _materialized_buffer_size(self) -> int:
        """Return materialized (compacted portion of) shuffle buffer size."""
        if self._shuffle_buffer is None:
            return 0
        return max(0, BlockAccessor.for_block(self._shuffle_buffer).num_rows() - self._batch_head)

    def next_batch(self) -> Block:
        """Get the next shuffled batch from the shuffle buffer.

        Returns:
            A batch represented as a Block.
        """
        assert self.has_batch() or (self._done_adding and self.has_any())
        if self._builder.num_rows() > 0 and (self._done_adding or self._materialized_buffer_size() <= self._buffer_min_size):
            if self._shuffle_buffer is not None:
                if self._batch_head > 0:
                    block = BlockAccessor.for_block(self._shuffle_buffer)
                    self._shuffle_buffer = block.slice(self._batch_head, block.num_rows())
                self._builder.add_block(self._shuffle_buffer)
            self._shuffle_buffer = self._builder.build()
            self._shuffle_buffer = BlockAccessor.for_block(self._shuffle_buffer).random_shuffle(self._shuffle_seed)
            if self._shuffle_seed is not None:
                self._shuffle_seed += 1
            if isinstance(BlockAccessor.for_block(self._shuffle_buffer), ArrowBlockAccessor) and self._shuffle_buffer.num_columns > 0 and (self._shuffle_buffer.column(0).num_chunks >= MIN_NUM_CHUNKS_TO_TRIGGER_COMBINE_CHUNKS):
                self._shuffle_buffer = transform_pyarrow.combine_chunks(self._shuffle_buffer)
            self._builder = DelegatingBlockBuilder()
            self._batch_head = 0
        assert self._shuffle_buffer is not None
        buffer_size = BlockAccessor.for_block(self._shuffle_buffer).num_rows()
        batch_size = min(self._batch_size, buffer_size)
        slice_start = self._batch_head
        self._batch_head += batch_size
        return BlockAccessor.for_block(self._shuffle_buffer).slice(slice_start, self._batch_head)