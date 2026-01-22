from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from .. import exc
from .. import util
from ..inspection import Inspectable
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypeAlias
def NotNullable(val: Union[_TypedColumnClauseArgument[Optional[_T]], Optional[Type[_T]]]) -> _TypedColumnClauseArgument[_T]:
    """Types a column or ORM class as not nullable.

    This can be used in select and other contexts to express that the value of
    a column cannot be null, for example due to a where condition on a
    nullable column::

        stmt = select(NotNullable(A.value)).where(A.value.is_not(None))

    At runtime this method returns the input unchanged.

    .. versionadded:: 2.0.20
    """
    return val