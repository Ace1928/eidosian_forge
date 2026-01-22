import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class QPURequest(Message):
    """
    Program and patch values to send to the QPU for execution.
    """
    program: Any
    'Execution settings and sequencer binaries.'
    patch_values: Dict[str, List[Any]]
    'Dictionary mapping data names to data values for patching the binary.'
    id: str
    'QPU request ID.'