from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
@property
def allocated(self) -> bool:
    return self.block_table is not None