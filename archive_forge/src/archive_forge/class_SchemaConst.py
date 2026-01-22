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
class SchemaConst(Enum):
    RETAIN_SCHEMA = 1
    'Symbol indicating that a :class:`_schema.Table`, :class:`.Sequence`\n    or in some cases a :class:`_schema.ForeignKey` object, in situations\n    where the object is being copied for a :meth:`.Table.to_metadata`\n    operation, should retain the schema name that it already has.\n\n    '
    BLANK_SCHEMA = 2
    "Symbol indicating that a :class:`_schema.Table` or :class:`.Sequence`\n    should have 'None' for its schema, even if the parent\n    :class:`_schema.MetaData` has specified a schema.\n\n    .. seealso::\n\n        :paramref:`_schema.MetaData.schema`\n\n        :paramref:`_schema.Table.schema`\n\n        :paramref:`.Sequence.schema`\n\n    "
    NULL_UNSPECIFIED = 3
    'Symbol indicating the "nullable" keyword was not passed to a Column.\n\n    This is used to distinguish between the use case of passing\n    ``nullable=None`` to a :class:`.Column`, which has special meaning\n    on some backends such as SQL Server.\n\n    '