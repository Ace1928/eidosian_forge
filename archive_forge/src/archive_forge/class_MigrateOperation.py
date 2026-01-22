from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
class MigrateOperation:
    """base class for migration command and organization objects.

    This system is part of the operation extensibility API.

    .. seealso::

        :ref:`operation_objects`

        :ref:`operation_plugins`

        :ref:`customizing_revision`

    """

    @util.memoized_property
    def info(self) -> Dict[Any, Any]:
        """A dictionary that may be used to store arbitrary information
        along with this :class:`.MigrateOperation` object.

        """
        return {}
    _mutations: FrozenSet[Rewriter] = frozenset()

    def reverse(self) -> MigrateOperation:
        raise NotImplementedError

    def to_diff_tuple(self) -> Tuple[Any, ...]:
        raise NotImplementedError