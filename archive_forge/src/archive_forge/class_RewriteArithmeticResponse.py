import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RewriteArithmeticResponse(Message):
    """
    The data needed to run programs with gate arithmetic on the hardware.
    """
    quil: str
    'Native Quil rewritten with no arithmetic in gate parameters.'
    original_memory_descriptors: Dict[str, ParameterSpec] = field(default_factory=dict)
    'The declared memory descriptors in the Quil of the related request.'
    recalculation_table: Dict[ParameterAref, str] = field(default_factory=dict)
    'A mapping from memory references to the original gate arithmetic.'