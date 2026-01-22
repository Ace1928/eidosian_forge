from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
@inspection._self_inspects
class SchemaItem(SchemaEventTarget, visitors.Visitable):
    """Base class for items that define a database schema."""
    __visit_name__ = 'schema_item'
    create_drop_stringify_dialect = 'default'

    def _init_items(self, *args: SchemaItem, **kw: Any) -> None:
        """Initialize the list of child items for this SchemaItem."""
        for item in args:
            if item is not None:
                try:
                    spwd = item._set_parent_with_dispatch
                except AttributeError as err:
                    raise exc.ArgumentError(f"'SchemaItem' object, such as a 'Column' or a 'Constraint' expected, got {item!r}") from err
                else:
                    spwd(self, **kw)

    def __repr__(self) -> str:
        return util.generic_repr(self, omit_kwarg=['info'])

    @util.memoized_property
    def info(self) -> _InfoType:
        """Info dictionary associated with the object, allowing user-defined
        data to be associated with this :class:`.SchemaItem`.

        The dictionary is automatically generated when first accessed.
        It can also be specified in the constructor of some objects,
        such as :class:`_schema.Table` and :class:`_schema.Column`.

        """
        return {}

    def _schema_item_copy(self, schema_item: _SI) -> _SI:
        if 'info' in self.__dict__:
            schema_item.info = self.info.copy()
        schema_item.dispatch._update(self.dispatch)
        return schema_item
    _use_schema_map = True