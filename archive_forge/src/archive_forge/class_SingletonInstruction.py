from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class SingletonInstruction(Instruction, _SingletonBase, overrides=_SingletonInstructionOverrides):
    """A base class to use for :class:`~.circuit.Instruction` objects that by default are singleton
    instances.

    This class should be used for instruction classes that have fixed definitions and do not contain
    any unique state. The canonical example of something like this is :class:`.Measure` which has an
    immutable definition and any instance of :class:`.Measure` is the same. Using singleton
    instructions as a base class for these types of gate classes provides a large advantage in the
    memory footprint of multiple instructions.

    The exception to be aware of with this class though are the :class:`~.circuit.Instruction`
    attributes :attr:`~.Instruction.label`, :attr:`~.Instruction.condition`,
    :attr:`~.Instruction.duration`, and :attr:`~.Instruction.unit` which can be set differently for
    specific instances of gates.  For :class:`SingletonInstruction` usage to be sound setting these
    attributes is not available and they can only be set at creation time, or on an object that has
    been specifically made mutable using :meth:`~.Instruction.to_mutable`. If any of these
    attributes are used during creation, then instead of using a single shared global instance of
    the same gate a new separate instance will be created."""
    __slots__ = ()