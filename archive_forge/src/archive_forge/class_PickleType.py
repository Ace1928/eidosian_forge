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
class PickleType(TypeDecorator[object]):
    """Holds Python objects, which are serialized using pickle.

    PickleType builds upon the Binary type to apply Python's
    ``pickle.dumps()`` to incoming objects, and ``pickle.loads()`` on
    the way out, allowing any pickleable Python object to be stored as
    a serialized binary field.

    To allow ORM change events to propagate for elements associated
    with :class:`.PickleType`, see :ref:`mutable_toplevel`.

    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, protocol: int=pickle.HIGHEST_PROTOCOL, pickler: Any=None, comparator: Optional[Callable[[Any, Any], bool]]=None, impl: Optional[_TypeEngineArgument[Any]]=None):
        """
        Construct a PickleType.

        :param protocol: defaults to ``pickle.HIGHEST_PROTOCOL``.

        :param pickler: defaults to pickle.  May be any object with
          pickle-compatible ``dumps`` and ``loads`` methods.

        :param comparator: a 2-arg callable predicate used
          to compare values of this type.  If left as ``None``,
          the Python "equals" operator is used to compare values.

        :param impl: A binary-storing :class:`_types.TypeEngine` class or
          instance to use in place of the default :class:`_types.LargeBinary`.
          For example the :class: `_mysql.LONGBLOB` class may be more effective
          when using MySQL.

          .. versionadded:: 1.4.20

        """
        self.protocol = protocol
        self.pickler = pickler or pickle
        self.comparator = comparator
        super().__init__()
        if impl:
            self.impl = to_instance(impl)

    def __reduce__(self):
        return (PickleType, (self.protocol, None, self.comparator))

    def bind_processor(self, dialect):
        impl_processor = self.impl_instance.bind_processor(dialect)
        dumps = self.pickler.dumps
        protocol = self.protocol
        if impl_processor:
            fixed_impl_processor = impl_processor

            def process(value):
                if value is not None:
                    value = dumps(value, protocol)
                return fixed_impl_processor(value)
        else:

            def process(value):
                if value is not None:
                    value = dumps(value, protocol)
                return value
        return process

    def result_processor(self, dialect, coltype):
        impl_processor = self.impl_instance.result_processor(dialect, coltype)
        loads = self.pickler.loads
        if impl_processor:
            fixed_impl_processor = impl_processor

            def process(value):
                value = fixed_impl_processor(value)
                if value is None:
                    return None
                return loads(value)
        else:

            def process(value):
                if value is None:
                    return None
                return loads(value)
        return process

    def compare_values(self, x, y):
        if self.comparator:
            return self.comparator(x, y)
        else:
            return x == y