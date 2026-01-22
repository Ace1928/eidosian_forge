from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
def _get_sentinel_column_for_table(self, table: Table) -> Optional[Sequence[Column[Any]]]:
    """given a :class:`.Table`, return a usable sentinel column or
        columns for this dialect if any.

        Return None if no sentinel columns could be identified, or raise an
        error if a column was marked as a sentinel explicitly but isn't
        compatible with this dialect.

        """
    sentinel_opts = self.dialect.insertmanyvalues_implicit_sentinel
    sentinel_characteristics = table._sentinel_column_characteristics
    sent_cols = sentinel_characteristics.columns
    if sent_cols is None:
        return None
    if sentinel_characteristics.is_autoinc:
        bitmask = self._sentinel_col_autoinc_lookup.get(sentinel_characteristics.default_characterization, 0)
    else:
        bitmask = self._sentinel_col_non_autoinc_lookup.get(sentinel_characteristics.default_characterization, 0)
    if sentinel_opts & bitmask:
        return sent_cols
    if sentinel_characteristics.is_explicit:
        raise exc.InvalidRequestError(f"Column {sent_cols[0]} can't be explicitly marked as a sentinel column when using the {self.dialect.name} dialect, as the particular type of default generation on this column is not currently compatible with this dialect's specific INSERT..RETURNING syntax which can receive the server-generated value in a deterministic way.  To remove this error, remove insert_sentinel=True from primary key autoincrement columns; these columns are automatically used as sentinels for supported dialects in any case.")
    return None