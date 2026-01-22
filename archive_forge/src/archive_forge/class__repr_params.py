from __future__ import annotations
from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class _repr_params(_repr_base):
    """Provide a string view of bound parameters.

    Truncates display to a given number of 'multi' parameter sets,
    as well as long values to a given number of characters.

    """
    __slots__ = ('params', 'batches', 'ismulti', 'max_params')

    def __init__(self, params: Optional[_AnyExecuteParams], batches: int, max_params: int=100, max_chars: int=300, ismulti: Optional[bool]=None):
        self.params = params
        self.ismulti = ismulti
        self.batches = batches
        self.max_chars = max_chars
        self.max_params = max_params

    def __repr__(self) -> str:
        if self.ismulti is None:
            return self.trunc(self.params)
        if isinstance(self.params, list):
            typ = self._LIST
        elif isinstance(self.params, tuple):
            typ = self._TUPLE
        elif isinstance(self.params, dict):
            typ = self._DICT
        else:
            return self.trunc(self.params)
        if self.ismulti:
            multi_params = cast('_AnyMultiExecuteParams', self.params)
            if len(self.params) > self.batches:
                msg = ' ... displaying %i of %i total bound parameter sets ... '
                return ' '.join((self._repr_multi(multi_params[:self.batches - 2], typ)[0:-1], msg % (self.batches, len(self.params)), self._repr_multi(multi_params[-2:], typ)[1:]))
            else:
                return self._repr_multi(multi_params, typ)
        else:
            return self._repr_params(cast('_AnySingleExecuteParams', self.params), typ)

    def _repr_multi(self, multi_params: _AnyMultiExecuteParams, typ: int) -> str:
        if multi_params:
            if isinstance(multi_params[0], list):
                elem_type = self._LIST
            elif isinstance(multi_params[0], tuple):
                elem_type = self._TUPLE
            elif isinstance(multi_params[0], dict):
                elem_type = self._DICT
            else:
                assert False, 'Unknown parameter type %s' % type(multi_params[0])
            elements = ', '.join((self._repr_params(params, elem_type) for params in multi_params))
        else:
            elements = ''
        if typ == self._LIST:
            return '[%s]' % elements
        else:
            return '(%s)' % elements

    def _get_batches(self, params: Iterable[Any]) -> Any:
        lparams = list(params)
        lenparams = len(lparams)
        if lenparams > self.max_params:
            lleft = self.max_params // 2
            return (lparams[0:lleft], lparams[-lleft:], lenparams - self.max_params)
        else:
            return (lparams, None, None)

    def _repr_params(self, params: _AnySingleExecuteParams, typ: int) -> str:
        if typ is self._DICT:
            return self._repr_param_dict(cast('_CoreSingleExecuteParams', params))
        elif typ is self._TUPLE:
            return self._repr_param_tuple(cast('Sequence[Any]', params))
        else:
            return self._repr_param_list(params)

    def _repr_param_dict(self, params: _CoreSingleExecuteParams) -> str:
        trunc = self.trunc
        items_first_batch, items_second_batch, trunclen = self._get_batches(params.items())
        if items_second_batch:
            text = '{%s' % ', '.join((f'{key!r}: {trunc(value)}' for key, value in items_first_batch))
            text += f' ... {trunclen} parameters truncated ... '
            text += '%s}' % ', '.join((f'{key!r}: {trunc(value)}' for key, value in items_second_batch))
        else:
            text = '{%s}' % ', '.join((f'{key!r}: {trunc(value)}' for key, value in items_first_batch))
        return text

    def _repr_param_tuple(self, params: Sequence[Any]) -> str:
        trunc = self.trunc
        items_first_batch, items_second_batch, trunclen = self._get_batches(params)
        if items_second_batch:
            text = '(%s' % ', '.join((trunc(value) for value in items_first_batch))
            text += f' ... {trunclen} parameters truncated ... '
            text += '%s)' % (', '.join((trunc(value) for value in items_second_batch)),)
        else:
            text = '(%s%s)' % (', '.join((trunc(value) for value in items_first_batch)), ',' if len(items_first_batch) == 1 else '')
        return text

    def _repr_param_list(self, params: _AnySingleExecuteParams) -> str:
        trunc = self.trunc
        items_first_batch, items_second_batch, trunclen = self._get_batches(params)
        if items_second_batch:
            text = '[%s' % ', '.join((trunc(value) for value in items_first_batch))
            text += f' ... {trunclen} parameters truncated ... '
            text += '%s]' % ', '.join((trunc(value) for value in items_second_batch))
        else:
            text = '[%s]' % ', '.join((trunc(value) for value in items_first_batch))
        return text