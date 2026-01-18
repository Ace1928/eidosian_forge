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
def visit_create_table(self, create, **kw):
    table = create.element
    preparer = self.preparer
    text = '\nCREATE '
    if table._prefixes:
        text += ' '.join(table._prefixes) + ' '
    text += 'TABLE '
    if create.if_not_exists:
        text += 'IF NOT EXISTS '
    text += preparer.format_table(table) + ' '
    create_table_suffix = self.create_table_suffix(table)
    if create_table_suffix:
        text += create_table_suffix + ' '
    text += '('
    separator = '\n'
    first_pk = False
    for create_column in create.columns:
        column = create_column.element
        try:
            processed = self.process(create_column, first_pk=column.primary_key and (not first_pk))
            if processed is not None:
                text += separator
                separator = ', \n'
                text += '\t' + processed
            if column.primary_key:
                first_pk = True
        except exc.CompileError as ce:
            raise exc.CompileError("(in table '%s', column '%s'): %s" % (table.description, column.name, ce.args[0])) from ce
    const = self.create_table_constraints(table, _include_foreign_key_constraints=create.include_foreign_key_constraints)
    if const:
        text += separator + '\t' + const
    text += '\n)%s\n\n' % self.post_create_table(table)
    return text