from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
def _impl_init_subclass(base: type[_SingletonBase], overrides: type[_SingletonInstructionOverrides]):

    def __init_subclass__(instruction_class, *, create_default_singleton=True, additional_singletons=(), **kwargs):
        super(base, instruction_class).__init_subclass__(**kwargs)
        if not create_default_singleton and (not additional_singletons):
            return
        instruction_class._singleton_static_lookup = {}

        class _Singleton(overrides, instruction_class, create_default_singleton=False):
            __module__ = None
            __slots__ = ()
            _singleton_init_arguments = {}

            def __new__(singleton_class, *_args, **_kwargs):
                raise TypeError(f"cannot create '{singleton_class.__name__}' instances")

            @property
            def base_class(self):
                return instruction_class

            @property
            def mutable(self):
                return False

            def to_mutable(self):
                args, kwargs = type(self)._singleton_init_arguments[id(self)]
                return self.base_class(*args, **kwargs, _force_mutable=True)

            def __setattr__(self, key, value):
                raise TypeError(f"This '{self.base_class.__name__}' object is immutable. You can get a mutable version by calling 'to_mutable()'.")

            def __copy__(self):
                return self

            def __deepcopy__(self, memo=None):
                return self

            def __reduce__(self):
                args, kwargs = type(self)._singleton_init_arguments[id(self)]
                return (functools.partial(instruction_class, **kwargs), args)
        _Singleton.__name__ = _Singleton.__qualname__ = f'_Singleton{instruction_class.__name__}'

        def _create_singleton_instance(args, kwargs):
            out = instruction_class(*args, **kwargs, _force_mutable=True)
            out = overrides._prepare_singleton_instance(out)
            out.__class__ = _Singleton
            _Singleton._singleton_init_arguments[id(out)] = (args, kwargs)
            key = instruction_class._singleton_lookup_key(*args, **kwargs)
            if key is not None:
                instruction_class._singleton_static_lookup[key] = out
            return out
        if create_default_singleton:
            instruction_class._singleton_default_instance = _create_singleton_instance((), {})
        for class_args, class_kwargs in additional_singletons:
            _create_singleton_instance(class_args, class_kwargs)
    return classmethod(__init_subclass__)