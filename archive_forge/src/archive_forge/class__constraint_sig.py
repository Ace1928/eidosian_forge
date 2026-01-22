from __future__ import annotations
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from typing_extensions import TypeGuard
from .. import util
from ..util import sqla_compat
class _constraint_sig(Generic[_C]):
    const: _C
    _sig: Tuple[Any, ...]
    name: Optional[sqla_compat._ConstraintNameDefined]
    impl: DefaultImpl
    _is_index: ClassVar[bool] = False
    _is_fk: ClassVar[bool] = False
    _is_uq: ClassVar[bool] = False
    _is_metadata: bool

    def __init_subclass__(cls) -> None:
        cls._register()

    @classmethod
    def _register(cls):
        raise NotImplementedError()

    def __init__(self, is_metadata: bool, impl: DefaultImpl, const: _C) -> None:
        raise NotImplementedError()

    def compare_to_reflected(self, other: _constraint_sig[Any]) -> ComparisonResult:
        assert self.impl is other.impl
        assert self._is_metadata
        assert not other._is_metadata
        return self._compare_to_reflected(other)

    def _compare_to_reflected(self, other: _constraint_sig[_C]) -> ComparisonResult:
        raise NotImplementedError()

    @classmethod
    def from_constraint(cls, is_metadata: bool, impl: DefaultImpl, constraint: _C) -> _constraint_sig[_C]:
        sig = _clsreg[constraint.__visit_name__](is_metadata, impl, constraint)
        return sig

    def md_name_to_sql_name(self, context: AutogenContext) -> Optional[str]:
        return sqla_compat._get_constraint_final_name(self.const, context.dialect)

    @util.memoized_property
    def is_named(self):
        return sqla_compat._constraint_is_named(self.const, self.impl.dialect)

    @util.memoized_property
    def unnamed(self) -> Tuple[Any, ...]:
        return self._sig

    @util.memoized_property
    def unnamed_no_options(self) -> Tuple[Any, ...]:
        raise NotImplementedError()

    @util.memoized_property
    def _full_sig(self) -> Tuple[Any, ...]:
        return (self.name,) + self.unnamed

    def __eq__(self, other) -> bool:
        return self._full_sig == other._full_sig

    def __ne__(self, other) -> bool:
        return self._full_sig != other._full_sig

    def __hash__(self) -> int:
        return hash(self._full_sig)