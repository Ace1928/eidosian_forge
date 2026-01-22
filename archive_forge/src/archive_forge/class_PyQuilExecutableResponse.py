import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class PyQuilExecutableResponse(Message):
    """
    rpcQ-serializable form of a pyQuil Program object.
    """
    program: str
    'String representation of a Quil program.'
    attributes: Dict[str, Any]
    'Miscellaneous attributes to be unpacked onto the pyQuil Program object.'