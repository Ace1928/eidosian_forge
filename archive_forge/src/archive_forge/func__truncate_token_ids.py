from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
    new_length = len(token_ids) // self.block_size * self.block_size
    return tuple(token_ids[:new_length])