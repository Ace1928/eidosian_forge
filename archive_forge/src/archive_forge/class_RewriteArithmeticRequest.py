import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RewriteArithmeticRequest(Message):
    """
    A request type to handle compiling arithmetic out of gate parameters.
    """
    quil: str
    'Native Quil for which to rewrite arithmetic parameters.'