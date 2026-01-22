from typing import List, Tuple
from ray.data._internal.block_list import BlockList
from ray.data._internal.split import _calculate_blocks_rows, _split_at_indices
from ray.data.block import Block, BlockMetadata, BlockPartition
from ray.types import ObjectRef
def _shave_all_splits(input_splits: List[BlockPartition], per_split_num_rows: List[List[int]], target_size: int) -> Tuple[List[BlockPartition], List[int], BlockPartition]:
    """Shave all block list to the target size.

    Args:
        input_splits: all block list to shave.
        input_splits: num rows (per block) for each block list.
        target_size: the upper bound target size of the shaved lists.
    Returns:
        A tuple of:
            - all shaved block list.
            - num of rows needed for the block list to meet the target size.
            - leftover blocks.
    """
    shaved_splits = []
    per_split_needed_rows = []
    leftovers = []
    for split, num_rows_per_block in zip(input_splits, per_split_num_rows):
        shaved, num_rows_needed, _leftovers = _shave_one_split(split, num_rows_per_block, target_size)
        shaved_splits.append(shaved)
        per_split_needed_rows.append(num_rows_needed)
        leftovers.extend(_leftovers)
    return (shaved_splits, per_split_needed_rows, leftovers)