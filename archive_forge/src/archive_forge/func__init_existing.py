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
def _init_existing(self, *args: Any, **kwargs: Any) -> None:
    autoload_with = kwargs.pop('autoload_with', None)
    autoload = kwargs.pop('autoload', autoload_with is not None)
    autoload_replace = kwargs.pop('autoload_replace', True)
    schema = kwargs.pop('schema', None)
    _extend_on = kwargs.pop('_extend_on', None)
    _reflect_info = kwargs.pop('_reflect_info', None)
    extend_existing = kwargs.pop('extend_existing', False)
    keep_existing = kwargs.pop('keep_existing', False)
    assert extend_existing
    assert not keep_existing
    if schema and schema != self.schema:
        raise exc.ArgumentError(f"Can't change schema of existing table from '{self.schema}' to '{schema}'")
    include_columns = kwargs.pop('include_columns', None)
    if include_columns is not None:
        for c in self.c:
            if c.name not in include_columns:
                self._columns.remove(c)
    resolve_fks = kwargs.pop('resolve_fks', True)
    for key in ('quote', 'quote_schema'):
        if key in kwargs:
            raise exc.ArgumentError("Can't redefine 'quote' or 'quote_schema' arguments")
    self.comment = kwargs.pop('comment', self.comment)
    self.implicit_returning = kwargs.pop('implicit_returning', self.implicit_returning)
    self.info = kwargs.pop('info', self.info)
    exclude_columns: _typing_Sequence[str]
    if autoload:
        if not autoload_replace:
            exclude_columns = [c.name for c in self.c]
        else:
            exclude_columns = ()
        self._autoload(self.metadata, autoload_with, include_columns, exclude_columns, resolve_fks, _extend_on=_extend_on, _reflect_info=_reflect_info)
    all_names = {c.name: c for c in self.c}
    self._extra_kwargs(**kwargs)
    self._init_items(*args, allow_replacements=True, all_names=all_names)