from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
@staticmethod
def _prepare_singleton_instance(instruction: Instruction):
    """Class-creation hook point.  Given an instance of the type that these overrides correspond
        to, this method should ensure that all lazy properties and caches that require mutation to
        write to are eagerly defined.

        Subclass "overrides" classes can override this method if the user/library-author-facing
        class they are providing overrides for has more lazy attributes or user-exposed state
        with interior mutability."""
    instruction._define()
    instruction._params = _frozenlist(instruction._params)
    return instruction