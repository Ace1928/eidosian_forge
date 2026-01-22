import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class AbstractKernel(Message):
    """
    An integration kernel defined for a specific frame. This abstract class is made concrete by either a `FilterKernel` or `TemplateKernel`
    """
    frame: str
    'The label of the associated rx-frame.'