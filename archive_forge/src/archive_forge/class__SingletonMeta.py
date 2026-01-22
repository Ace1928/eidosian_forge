from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class _SingletonMeta(type(Instruction)):
    __slots__ = ()

    def __new__(mcs, name, bases, namespace, *, overrides=None, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if overrides is not None:
            cls.__init_subclass__ = _impl_init_subclass(cls, overrides)
        return cls

    def __call__(cls, *args, _force_mutable=False, **kwargs):
        if _force_mutable:
            return super().__call__(*args, **kwargs)
        if not args and (not kwargs):
            return cls._singleton_default_instance
        if (key := cls._singleton_lookup_key(*args, **kwargs)) is not None:
            try:
                singleton = cls._singleton_static_lookup.get(key)
            except TypeError:
                singleton = None
            if singleton is not None:
                return singleton
        return super().__call__(*args, **kwargs)