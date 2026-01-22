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
class DropConstraint(_DropBase):
    """Represent an ALTER TABLE DROP CONSTRAINT statement."""
    __visit_name__ = 'drop_constraint'

    def __init__(self, element, cascade=False, if_exists=False, **kw):
        self.cascade = cascade
        super().__init__(element, if_exists=if_exists, **kw)
        element._create_rule = util.portable_instancemethod(self._create_rule_disable)