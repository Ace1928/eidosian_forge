from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
def _create_singleton_instance(args, kwargs):
    out = instruction_class(*args, **kwargs, _force_mutable=True)
    out = overrides._prepare_singleton_instance(out)
    out.__class__ = _Singleton
    _Singleton._singleton_init_arguments[id(out)] = (args, kwargs)
    key = instruction_class._singleton_lookup_key(*args, **kwargs)
    if key is not None:
        instruction_class._singleton_static_lookup[key] = out
    return out