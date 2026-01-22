from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
def _reject_mutation(self, *args, **kwargs):
    raise TypeError("'params' of singletons cannot be mutated")