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
class AddConstraintOp(MigrateOperation):
    """Represent an add constraint operation."""
    add_constraint_ops = util.Dispatcher()

    @property
    def constraint_type(self) -> str:
        raise NotImplementedError()

    @classmethod
    def register_add_constraint(cls, type_: str) -> Callable[[Type[_AC]], Type[_AC]]:

        def go(klass: Type[_AC]) -> Type[_AC]:
            cls.add_constraint_ops.dispatch_for(type_)(klass.from_constraint)
            return klass
        return go

    @classmethod
    def from_constraint(cls, constraint: Constraint) -> AddConstraintOp:
        return cls.add_constraint_ops.dispatch(constraint.__visit_name__)(constraint)

    @abstractmethod
    def to_constraint(self, migration_context: Optional[MigrationContext]=None) -> Constraint:
        pass

    def reverse(self) -> DropConstraintOp:
        return DropConstraintOp.from_constraint(self.to_constraint())

    def to_diff_tuple(self) -> Tuple[str, Constraint]:
        return ('add_constraint', self.to_constraint())