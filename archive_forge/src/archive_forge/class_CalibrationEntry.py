from __future__ import annotations
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence, Callable
from enum import IntEnum
from typing import Any
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj.pulse_qobj import PulseQobjInstruction
from qiskit.exceptions import QiskitError
class CalibrationEntry(metaclass=ABCMeta):
    """A metaclass of a calibration entry.

    This class defines a standard model of Qiskit pulse program that is
    agnostic to the underlying in-memory representation.

    This entry distinguishes whether this is provided by end-users or a backend
    by :attr:`.user_provided` attribute which may be provided when
    the actual calibration data is provided to the entry with by :meth:`define`.

    Note that a custom entry provided by an end-user may appear in the wire-format
    as an inline calibration, e.g. :code:`defcal` of the QASM3,
    that may update the backend instruction set architecture for execution.

    .. note::

        This and built-in subclasses are expected to be private without stable user-facing API.
        The purpose of this class is to wrap different
        in-memory pulse program representations in Qiskit, so that it can provide
        the standard data model and API which are primarily used by the transpiler ecosystem.
        It is assumed that end-users will never directly instantiate this class,
        but :class:`.Target` or :class:`.InstructionScheduleMap` internally use this data model
        to avoid implementing a complicated branching logic to
        manage different calibration data formats.

    """

    @abstractmethod
    def define(self, definition: Any, user_provided: bool):
        """Attach definition to the calibration entry.

        Args:
            definition: Definition of this entry.
            user_provided: If this entry is defined by user.
                If the flag is set, this calibration may appear in the wire format
                as an inline calibration, to override the backend instruction set architecture.
        """
        pass

    @abstractmethod
    def get_signature(self) -> inspect.Signature:
        """Return signature object associated with entry definition.

        Returns:
            Signature object.
        """
        pass

    @abstractmethod
    def get_schedule(self, *args, **kwargs) -> Schedule | ScheduleBlock:
        """Generate schedule from entry definition.

        If the pulse program is templated with :class:`.Parameter` objects,
        you can provide corresponding parameter values for this method
        to get a particular pulse program with assigned parameters.

        Args:
            args: Command parameters.
            kwargs: Command keyword parameters.

        Returns:
            Pulse schedule with assigned parameters.
        """
        pass

    @property
    @abstractmethod
    def user_provided(self) -> bool:
        """Return if this entry is user defined."""
        pass