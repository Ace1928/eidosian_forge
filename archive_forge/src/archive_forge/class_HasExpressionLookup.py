from __future__ import annotations
import collections.abc as collections_abc
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from uuid import UUID as _python_UUID
from . import coercions
from . import elements
from . import operators
from . import roles
from . import type_api
from .base import _NONE_NAME
from .base import NO_ARG
from .base import SchemaEventTarget
from .cache_key import HasCacheKey
from .elements import quoted_name
from .elements import Slice
from .elements import TypeCoerce as type_coerce  # noqa
from .type_api import Emulated
from .type_api import NativeForEmulated  # noqa
from .type_api import to_instance as to_instance
from .type_api import TypeDecorator as TypeDecorator
from .type_api import TypeEngine as TypeEngine
from .type_api import TypeEngineMixin
from .type_api import Variant  # noqa
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..engine import processors
from ..util import langhelpers
from ..util import OrderedDict
from ..util.typing import is_literal
from ..util.typing import Literal
from ..util.typing import typing_get_args
class HasExpressionLookup(TypeEngineMixin):
    """Mixin expression adaptations based on lookup tables.

    These rules are currently used by the numeric, integer and date types
    which have detailed cross-expression coercion rules.

    """

    @property
    def _expression_adaptations(self):
        raise NotImplementedError()

    class Comparator(TypeEngine.Comparator[_CT]):
        __slots__ = ()
        _blank_dict = util.EMPTY_DICT

        def _adapt_expression(self, op: OperatorType, other_comparator: TypeEngine.Comparator[Any]) -> Tuple[OperatorType, TypeEngine[Any]]:
            othertype = other_comparator.type._type_affinity
            if TYPE_CHECKING:
                assert isinstance(self.type, HasExpressionLookup)
            lookup = self.type._expression_adaptations.get(op, self._blank_dict).get(othertype, self.type)
            if lookup is othertype:
                return (op, other_comparator.type)
            elif lookup is self.type._type_affinity:
                return (op, self.type)
            else:
                return (op, to_instance(lookup))
    comparator_factory: _ComparatorFactory[Any] = Comparator