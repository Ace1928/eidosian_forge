from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class LambdaElement(elements.ClauseElement):
    """A SQL construct where the state is stored as an un-invoked lambda.

    The :class:`_sql.LambdaElement` is produced transparently whenever
    passing lambda expressions into SQL constructs, such as::

        stmt = select(table).where(lambda: table.c.col == parameter)

    The :class:`_sql.LambdaElement` is the base of the
    :class:`_sql.StatementLambdaElement` which represents a full statement
    within a lambda.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`engine_lambda_caching`

    """
    __visit_name__ = 'lambda_element'
    _is_lambda_element = True
    _traverse_internals = [('_resolved', visitors.InternalTraversal.dp_clauseelement)]
    _transforms: Tuple[_CloneCallableType, ...] = ()
    _resolved_bindparams: List[BindParameter[Any]]
    parent_lambda: Optional[StatementLambdaElement] = None
    closure_cache_key: Union[Tuple[Any, ...], Literal[CacheConst.NO_CACHE]]
    role: Type[SQLRole]
    _rec: Union[AnalyzedFunction, NonAnalyzedFunction]
    fn: _AnyLambdaType
    tracker_key: Tuple[CodeType, ...]

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.fn.__code__)

    def __init__(self, fn: _LambdaType, role: Type[SQLRole], opts: Union[Type[LambdaOptions], LambdaOptions]=LambdaOptions, apply_propagate_attrs: Optional[ClauseElement]=None):
        self.fn = fn
        self.role = role
        self.tracker_key = (fn.__code__,)
        self.opts = opts
        if apply_propagate_attrs is None and role is roles.StatementRole:
            apply_propagate_attrs = self
        rec = self._retrieve_tracker_rec(fn, apply_propagate_attrs, opts)
        if apply_propagate_attrs is not None:
            propagate_attrs = rec.propagate_attrs
            if propagate_attrs:
                apply_propagate_attrs._propagate_attrs = propagate_attrs

    def _retrieve_tracker_rec(self, fn, apply_propagate_attrs, opts):
        lambda_cache = opts.lambda_cache
        if lambda_cache is None:
            lambda_cache = _closure_per_cache_key
        tracker_key = self.tracker_key
        fn = self.fn
        closure = fn.__closure__
        tracker = AnalyzedCode.get(fn, self, opts)
        bindparams: List[BindParameter[Any]]
        self._resolved_bindparams = bindparams = []
        if self.parent_lambda is not None:
            parent_closure_cache_key = self.parent_lambda.closure_cache_key
        else:
            parent_closure_cache_key = ()
        cache_key: Union[Tuple[Any, ...], Literal[CacheConst.NO_CACHE]]
        if parent_closure_cache_key is not _cache_key.NO_CACHE:
            anon_map = visitors.anon_map()
            cache_key = tuple([getter(closure, opts, anon_map, bindparams) for getter in tracker.closure_trackers])
            if _cache_key.NO_CACHE not in anon_map:
                cache_key = parent_closure_cache_key + cache_key
                self.closure_cache_key = cache_key
                try:
                    rec = lambda_cache[tracker_key + cache_key]
                except KeyError:
                    rec = None
            else:
                cache_key = _cache_key.NO_CACHE
                rec = None
        else:
            cache_key = _cache_key.NO_CACHE
            rec = None
        self.closure_cache_key = cache_key
        if rec is None:
            if cache_key is not _cache_key.NO_CACHE:
                with AnalyzedCode._generation_mutex:
                    key = tracker_key + cache_key
                    if key not in lambda_cache:
                        rec = AnalyzedFunction(tracker, self, apply_propagate_attrs, fn)
                        rec.closure_bindparams = bindparams
                        lambda_cache[key] = rec
                    else:
                        rec = lambda_cache[key]
            else:
                rec = NonAnalyzedFunction(self._invoke_user_fn(fn))
        else:
            bindparams[:] = [orig_bind._with_value(new_bind.value, maintain_key=True) for orig_bind, new_bind in zip(rec.closure_bindparams, bindparams)]
        self._rec = rec
        if cache_key is not _cache_key.NO_CACHE:
            if self.parent_lambda is not None:
                bindparams[:0] = self.parent_lambda._resolved_bindparams
            lambda_element: Optional[LambdaElement] = self
            while lambda_element is not None:
                rec = lambda_element._rec
                if rec.bindparam_trackers:
                    tracker_instrumented_fn = rec.tracker_instrumented_fn
                    for tracker in rec.bindparam_trackers:
                        tracker(lambda_element.fn, tracker_instrumented_fn, bindparams)
                lambda_element = lambda_element.parent_lambda
        return rec

    def __getattr__(self, key):
        return getattr(self._rec.expected_expr, key)

    @property
    def _is_sequence(self):
        return self._rec.is_sequence

    @property
    def _select_iterable(self):
        if self._is_sequence:
            return itertools.chain.from_iterable([element._select_iterable for element in self._resolved])
        else:
            return self._resolved._select_iterable

    @property
    def _from_objects(self):
        if self._is_sequence:
            return itertools.chain.from_iterable([element._from_objects for element in self._resolved])
        else:
            return self._resolved._from_objects

    def _param_dict(self):
        return {b.key: b.value for b in self._resolved_bindparams}

    def _setup_binds_for_tracked_expr(self, expr):
        bindparam_lookup = {b.key: b for b in self._resolved_bindparams}

        def replace(element: Optional[visitors.ExternallyTraversible], **kw: Any) -> Optional[visitors.ExternallyTraversible]:
            if isinstance(element, elements.BindParameter):
                if element.key in bindparam_lookup:
                    bind = bindparam_lookup[element.key]
                    if element.expanding:
                        bind.expanding = True
                        bind.expand_op = element.expand_op
                        bind.type = element.type
                    return bind
            return None
        if self._rec.is_sequence:
            expr = [visitors.replacement_traverse(sub_expr, {}, replace) for sub_expr in expr]
        elif getattr(expr, 'is_clause_element', False):
            expr = visitors.replacement_traverse(expr, {}, replace)
        return expr

    def _copy_internals(self, clone: _CloneCallableType=_clone, deferred_copy_internals: Optional[_CloneCallableType]=None, **kw: Any) -> None:
        self._resolved = clone(self._resolved, deferred_copy_internals=deferred_copy_internals, **kw)

    @util.memoized_property
    def _resolved(self):
        expr = self._rec.expected_expr
        if self._resolved_bindparams:
            expr = self._setup_binds_for_tracked_expr(expr)
        return expr

    def _gen_cache_key(self, anon_map, bindparams):
        if self.closure_cache_key is _cache_key.NO_CACHE:
            anon_map[_cache_key.NO_CACHE] = True
            return None
        cache_key = (self.fn.__code__, self.__class__) + self.closure_cache_key
        parent = self.parent_lambda
        while parent is not None:
            assert parent.closure_cache_key is not CacheConst.NO_CACHE
            parent_closure_cache_key: Tuple[Any, ...] = parent.closure_cache_key
            cache_key = (parent.fn.__code__,) + parent_closure_cache_key + cache_key
            parent = parent.parent_lambda
        if self._resolved_bindparams:
            bindparams.extend(self._resolved_bindparams)
        return cache_key

    def _invoke_user_fn(self, fn: _AnyLambdaType, *arg: Any) -> ClauseElement:
        return fn()