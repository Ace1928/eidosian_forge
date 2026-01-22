import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
    blocks: Set[PhysicalTokenBlock] = set()
    for seq in seq_group.get_seqs():
        if seq.is_finished():
            continue
        blocks.update(self.block_tables[seq.seq_id])
    return list(blocks)