import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class BinaryExecutableResponse(Message):
    """
    Program to run on the QPU.
    """
    program: str
    'Execution settings and sequencer binaries.'
    memory_descriptors: Dict[str, ParameterSpec] = field(default_factory=dict)
    'Internal field for constructing patch tables.'
    ro_sources: List[Any] = field(default_factory=list)
    'Internal field for reshaping returned buffers.'