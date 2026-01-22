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
@util.preload_module('orm.properties')
def _setup_arguments_on_columns(self) -> None:
    """Propagate configuration arguments made on this composite
        to the target columns, for those that apply.

        """
    ColumnProperty = util.preloaded.orm_properties.ColumnProperty
    for prop in self.props:
        if not isinstance(prop, ColumnProperty):
            continue
        else:
            cprop = prop
        cprop.active_history = self.active_history
        if self.deferred:
            cprop.deferred = self.deferred
            cprop.strategy_key = (('deferred', True), ('instrument', True))
        cprop.group = self.group