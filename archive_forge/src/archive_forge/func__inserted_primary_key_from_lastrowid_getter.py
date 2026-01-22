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
@util.memoized_property
@util.preload_module('sqlalchemy.engine.result')
def _inserted_primary_key_from_lastrowid_getter(self):
    result = util.preloaded.engine_result
    param_key_getter = self._within_exec_param_key_getter
    assert self.compile_state is not None
    statement = self.compile_state.statement
    if TYPE_CHECKING:
        assert isinstance(statement, Insert)
    table = statement.table
    getters = [(operator.methodcaller('get', param_key_getter(col), None), col) for col in table.primary_key]
    autoinc_getter = None
    autoinc_col = table._autoincrement_column
    if autoinc_col is not None:
        lastrowid_processor = autoinc_col.type._cached_result_processor(self.dialect, None)
        autoinc_key = param_key_getter(autoinc_col)
        if autoinc_key in self.binds:

            def _autoinc_getter(lastrowid, parameters):
                param_value = parameters.get(autoinc_key, lastrowid)
                if param_value is not None:
                    return param_value
                else:
                    return lastrowid
            autoinc_getter = _autoinc_getter
    else:
        lastrowid_processor = None
    row_fn = result.result_tuple([col.key for col in table.primary_key])

    def get(lastrowid, parameters):
        """given cursor.lastrowid value and the parameters used for INSERT,
            return a "row" that represents the primary key, either by
            using the "lastrowid" or by extracting values from the parameters
            that were sent along with the INSERT.

            """
        if lastrowid_processor is not None:
            lastrowid = lastrowid_processor(lastrowid)
        if lastrowid is None:
            return row_fn((getter(parameters) for getter, col in getters))
        else:
            return row_fn(((autoinc_getter(lastrowid, parameters) if autoinc_getter is not None else lastrowid) if col is autoinc_col else getter(parameters) for getter, col in getters))
    return get