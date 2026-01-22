import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class ConjugateByCliffordResponse(Message):
    """
    RPC reply payload for a Pauli element as conjugated by a Clifford element.
    """
    phase: int
    'Encoded global phase factor on the emitted Pauli.'
    pauli: str
    'Description of the encoded Pauli.'