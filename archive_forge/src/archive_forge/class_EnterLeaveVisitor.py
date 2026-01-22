from copy import copy
from enum import Enum
from typing import (
from ..pyutils import inspect, snake_to_camel
from . import ast
from .ast import Node, QUERY_DOCUMENT_KEYS
class EnterLeaveVisitor(NamedTuple):
    """Visitor with functions for entering and leaving."""
    enter: Optional[Callable[..., Optional[VisitorAction]]]
    leave: Optional[Callable[..., Optional[VisitorAction]]]