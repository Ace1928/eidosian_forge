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
def visit_override_binds(self, override_binds, **kw):
    """SQL compile the nested element of an _OverrideBinds with
        bindparams swapped out.

        The _OverrideBinds is not normally expected to be compiled; it
        is meant to be used when an already cached statement is to be used,
        the compilation was already performed, and only the bound params should
        be swapped in at execution time.

        However, there are test cases that exericise this object, and
        additionally the ORM subquery loader is known to feed in expressions
        which include this construct into new queries (discovered in #11173),
        so it has to do the right thing at compile time as well.

        """
    sqltext = override_binds.element._compiler_dispatch(self, **kw)
    for k in override_binds.translate:
        if k not in self.binds:
            continue
        bp = self.binds[k]
        new_bp = bp._with_value(override_binds.translate[bp.key], maintain_key=True, required=False)
        name = self.bind_names[bp]
        self.binds[k] = self.binds[name] = new_bp
        self.bind_names[new_bp] = name
        self.bind_names.pop(bp, None)
        if bp in self.post_compile_params:
            self.post_compile_params |= {new_bp}
        if bp in self.literal_execute_params:
            self.literal_execute_params |= {new_bp}
        ckbm_tuple = self._cache_key_bind_match
        if ckbm_tuple:
            ckbm, cksm = ckbm_tuple
            for bp in bp._cloned_set:
                if bp.key in cksm:
                    cb = cksm[bp.key]
                    ckbm[cb].append(new_bp)
    return sqltext