from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class WriteOnlyMapped(_MappedAnnotationBase[_T_co]):
    """Represent the ORM mapped attribute type for a "write only" relationship.

    The :class:`_orm.WriteOnlyMapped` type annotation may be used in an
    :ref:`Annotated Declarative Table <orm_declarative_mapped_column>` mapping
    to indicate that the ``lazy="write_only"`` loader strategy should be used
    for a particular :func:`_orm.relationship`.

    E.g.::

        class User(Base):
            __tablename__ = "user"
            id: Mapped[int] = mapped_column(primary_key=True)
            addresses: WriteOnlyMapped[Address] = relationship(
                cascade="all,delete-orphan"
            )

    See the section :ref:`write_only_relationship` for background.

    .. versionadded:: 2.0

    .. seealso::

        :ref:`write_only_relationship` - complete background

        :class:`.DynamicMapped` - includes legacy :class:`_orm.Query` support

    """
    __slots__ = ()
    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: Any) -> InstrumentedAttribute[_T_co]:
            ...

        @overload
        def __get__(self, instance: object, owner: Any) -> WriteOnlyCollection[_T_co]:
            ...

        def __get__(self, instance: Optional[object], owner: Any) -> Union[InstrumentedAttribute[_T_co], WriteOnlyCollection[_T_co]]:
            ...

        def __set__(self, instance: Any, value: typing.Collection[_T_co]) -> None:
            ...