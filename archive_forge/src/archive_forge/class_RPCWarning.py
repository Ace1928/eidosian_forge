import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RPCWarning(Message):
    """
    An individual warning emitted in the course of RPC processing.
    """
    body: str
    'The warning string.'
    kind: Optional[str] = None
    'The type of the warning raised.'