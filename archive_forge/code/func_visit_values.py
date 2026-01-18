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
def visit_values(self, element, asfrom=False, from_linter=None, **kw):
    v = self._render_values(element, **kw)
    if element._unnamed:
        name = None
    elif isinstance(element.name, elements._truncated_label):
        name = self._truncated_identifier('values', element.name)
    else:
        name = element.name
    if element._is_lateral:
        lateral = 'LATERAL '
    else:
        lateral = ''
    if asfrom:
        if from_linter:
            from_linter.froms[element._de_clone()] = name if name is not None else '(unnamed VALUES element)'
        if name:
            kw['include_table'] = False
            v = '%s(%s)%s (%s)' % (lateral, v, self.get_render_as_alias_suffix(self.preparer.quote(name)), ', '.join((c._compiler_dispatch(self, **kw) for c in element.columns)))
        else:
            v = '%s(%s)' % (lateral, v)
    return v