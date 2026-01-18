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
def visit_textual_select(self, taf, compound_index=None, asfrom=False, **kw):
    toplevel = not self.stack
    entry = self._default_stack_entry if toplevel else self.stack[-1]
    new_entry: _CompilerStackEntry = {'correlate_froms': set(), 'asfrom_froms': set(), 'selectable': taf}
    self.stack.append(new_entry)
    if taf._independent_ctes:
        self._dispatch_independent_ctes(taf, kw)
    populate_result_map = toplevel or (compound_index == 0 and entry.get('need_result_map_for_compound', False)) or entry.get('need_result_map_for_nested', False)
    if populate_result_map:
        self._ordered_columns = self._textual_ordered_columns = taf.positional
        self._loose_column_name_matching = not taf.positional and bool(taf.column_args)
        for c in taf.column_args:
            self.process(c, within_columns_clause=True, add_to_result_map=self._add_to_result_map)
    text = self.process(taf.element, **kw)
    if self.ctes:
        nesting_level = len(self.stack) if not toplevel else None
        text = self._render_cte_clause(nesting_level=nesting_level) + text
    self.stack.pop(-1)
    return text