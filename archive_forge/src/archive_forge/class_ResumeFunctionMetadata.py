import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: List[Instruction] = dataclasses.field(default_factory=list)
    prefix_block_target_offset_remap: List[int] = dataclasses.field(default_factory=list)
    block_target_offset_remap: Optional[Dict[int, int]] = None