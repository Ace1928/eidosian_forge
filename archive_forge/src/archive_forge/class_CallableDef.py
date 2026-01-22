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
class CallableDef(CalibrationEntry):
    """Python callback function that generates Qiskit Pulse program.

    A callable is inspected by the python built-in inspection module and
    provide the signature. This entry is parameterized by the function signature
    and .get_schedule method returns a non-parameterized pulse program
    by consuming the provided arguments and keyword arguments.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self):
        """Define an empty entry."""
        self._definition = None
        self._signature = None
        self._user_provided = None

    @property
    def user_provided(self) -> bool:
        return self._user_provided

    def define(self, definition: Callable, user_provided: bool=True):
        self._definition = definition
        self._signature = inspect.signature(definition)
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        return self._signature

    def get_schedule(self, *args, **kwargs) -> Schedule | ScheduleBlock:
        try:
            to_bind = self._signature.bind(*args, **kwargs)
            to_bind.apply_defaults()
        except TypeError as ex:
            raise PulseError("Assigned parameter doesn't match with function signature.") from ex
        out = self._definition(**to_bind.arguments)
        if 'publisher' not in out.metadata:
            if self.user_provided:
                out.metadata['publisher'] = CalibrationPublisher.QISKIT
            else:
                out.metadata['publisher'] = CalibrationPublisher.BACKEND_PROVIDER
        return out

    def __eq__(self, other):
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        params_str = ', '.join(self.get_signature().parameters.keys())
        return f'Callable {self._definition.__name__}({params_str})'