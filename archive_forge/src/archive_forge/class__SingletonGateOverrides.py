from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class _SingletonGateOverrides(_SingletonInstructionOverrides, Gate):
    """Overrides for all the mutable methods and properties of `Gate` to make it immutable.

    This class just exists for the principle; there's no additional overrides required compared
    to :class:`~.circuit.Instruction`."""
    __slots__ = ()