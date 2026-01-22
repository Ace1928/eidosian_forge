from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
@staticmethod
def _push_right_prepend(this: Schedule, other: ScheduleComponent) -> Schedule:
    """Return ``this`` with ``other`` inserted at the latest possible time
        such that ``other`` ends before it overlaps with any of ``this``.

        If required ``this`` is shifted  to start late enough so that there is room
        to insert ``other``.

        Args:
           this: Input schedule to which ``other`` will be inserted.
           other: Other schedule to insert.

        Returns:
           Push right prepended schedule.
        """
    this_channels = set(this.channels)
    other_channels = set(other.channels)
    shared_channels = list(this_channels & other_channels)
    ch_slacks = [this.ch_start_time(channel) - other.ch_stop_time(channel) for channel in shared_channels]
    if ch_slacks:
        insert_time = min(ch_slacks) + other.start_time
    else:
        insert_time = this.stop_time - other.stop_time + other.start_time
    if insert_time < 0:
        this.shift(-insert_time, inplace=True)
        this.insert(0, other, inplace=True)
    else:
        this.insert(insert_time, other, inplace=True)
    return this