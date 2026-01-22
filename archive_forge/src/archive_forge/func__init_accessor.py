from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
def _init_accessor(self) -> None:
    if is_dataclass(self.composite_class) and (not hasattr(self.composite_class, '__composite_values__')):
        insp = inspect.signature(self.composite_class)
        getter = operator.attrgetter(*[p.name for p in insp.parameters.values()])
        if len(insp.parameters) == 1:
            self._generated_composite_accessor = lambda obj: (getter(obj),)
        else:
            self._generated_composite_accessor = getter
    if self.composite_class is not None and isinstance(self.composite_class, type) and (self.composite_class not in _composite_getters):
        if self._generated_composite_accessor is not None:
            _composite_getters[self.composite_class] = self._generated_composite_accessor
        elif hasattr(self.composite_class, '__composite_values__'):
            _composite_getters[self.composite_class] = lambda obj: obj.__composite_values__()