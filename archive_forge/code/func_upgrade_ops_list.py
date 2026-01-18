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
@property
def upgrade_ops_list(self) -> List[UpgradeOps]:
    """A list of :class:`.UpgradeOps` instances.

        This is used in place of the :attr:`.MigrationScript.upgrade_ops`
        attribute when dealing with a revision operation that does
        multiple autogenerate passes.

        """
    return self._upgrade_ops