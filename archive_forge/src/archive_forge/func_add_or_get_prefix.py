from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
def add_or_get_prefix(self, token_ids: Sequence[int], lora_int_id: int) -> Optional[Prefix]:
    token_ids = self._truncate_token_ids(token_ids)
    if len(token_ids) == 0:
        return None
    prefix = Prefix(token_ids, self.block_size)
    prefix_hash = hash((prefix, lora_int_id))
    if prefix_hash not in self.prefixes:
        self.prefixes[prefix_hash] = prefix
    return self.prefixes[prefix_hash]