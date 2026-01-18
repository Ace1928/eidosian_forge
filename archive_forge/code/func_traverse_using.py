from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def traverse_using(iterator: Iterable[ExternallyTraversible], obj: Optional[ExternallyTraversible], visitors: Mapping[str, _TraverseCallableType[Any]]) -> Optional[ExternallyTraversible]:
    """Visit the given expression structure using the given iterator of
    objects.

    :func:`.visitors.traverse_using` is usually called internally as the result
    of the :func:`.visitors.traverse` function.

    :param iterator: an iterable or sequence which will yield
     :class:`_expression.ClauseElement`
     structures; the iterator is assumed to be the
     product of the :func:`.visitors.iterate` function.

    :param obj: the :class:`_expression.ClauseElement`
     that was used as the target of the
     :func:`.iterate` function.

    :param visitors: dictionary of visit functions.  See :func:`.traverse`
     for details on this dictionary.

    .. seealso::

        :func:`.traverse`


    """
    for target in iterator:
        meth = visitors.get(target.__visit_name__, None)
        if meth:
            meth(target)
    return obj