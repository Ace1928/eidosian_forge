import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_getattr
class SetAttribute(AttributeTemplate):
    key = types.Set

    @bound_function('set.add')
    def resolve_add(self, set, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(set.dtype, item)
        if unified is not None:
            sig = signature(types.none, unified)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    @bound_function('set.update')
    def resolve_update(self, set, args, kws):
        iterable, = args
        assert not kws
        if not isinstance(iterable, types.IterableType):
            return
        dtype = iterable.iterator_type.yield_type
        unified = self.context.unify_pairs(set.dtype, dtype)
        if unified is not None:
            sig = signature(types.none, iterable)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    def _resolve_operator(self, set, args, kws):
        assert not kws
        iterable, = args
        if isinstance(iterable, types.Set) and iterable.dtype == set.dtype:
            return signature(set, iterable)

    def _resolve_comparator(self, set, args, kws):
        assert not kws
        arg, = args
        if arg == set:
            return signature(types.boolean, arg)