from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
class DropSequence(_DropBase):
    """Represent a DROP SEQUENCE statement."""
    __visit_name__ = 'drop_sequence'

    def __init__(self, element: Sequence, if_exists: bool=False):
        super().__init__(element, if_exists=if_exists)