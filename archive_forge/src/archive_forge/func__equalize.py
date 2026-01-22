from typing import List, Tuple
from ray.data._internal.block_list import BlockList
from ray.data._internal.split import _calculate_blocks_rows, _split_at_indices
from ray.data.block import Block, BlockMetadata, BlockPartition
from ray.types import ObjectRef
def _equalize(per_split_block_lists: List[BlockList], owned_by_consumer: bool) -> List[BlockList]:
    """Equalize split block lists into equal number of rows.

    Args:
        per_split_block_lists: block lists to equalize.
    Returns:
        the equalized block lists.
    """
    if len(per_split_block_lists) == 0:
        return per_split_block_lists
    per_split_blocks_with_metadata = [block_list.get_blocks_with_metadata() for block_list in per_split_block_lists]
    per_split_num_rows: List[List[int]] = [_calculate_blocks_rows(split) for split in per_split_blocks_with_metadata]
    total_rows = sum([sum(blocks_rows) for blocks_rows in per_split_num_rows])
    target_split_size = total_rows // len(per_split_blocks_with_metadata)
    shaved_splits, per_split_needed_rows, leftovers = _shave_all_splits(per_split_blocks_with_metadata, per_split_num_rows, target_split_size)
    for shaved_split, split_needed_row in zip(shaved_splits, per_split_needed_rows):
        num_shaved_rows = sum([meta.num_rows for _, meta in shaved_split])
        assert num_shaved_rows <= target_split_size
        assert num_shaved_rows + split_needed_row == target_split_size
    leftover_refs = []
    leftover_meta = []
    for ref, meta in leftovers:
        leftover_refs.append(ref)
        leftover_meta.append(meta)
    leftover_splits = _split_leftovers(BlockList(leftover_refs, leftover_meta, owned_by_consumer=owned_by_consumer), per_split_needed_rows)
    for i, leftover_split in enumerate(leftover_splits):
        shaved_splits[i].extend(leftover_split)
        num_shaved_rows = sum([meta.num_rows for _, meta in shaved_splits[i]])
        assert num_shaved_rows == target_split_size
    equalized_block_lists: List[BlockList] = []
    for split in shaved_splits:
        block_refs: List[ObjectRef[Block]] = []
        meta: List[BlockMetadata] = []
        for block_ref, m in split:
            block_refs.append(block_ref)
            meta.append(m)
        equalized_block_lists.append(BlockList(block_refs, meta, owned_by_consumer=owned_by_consumer))
    return equalized_block_lists