from __future__ import annotations
import abc
from functools import singledispatch
from collections.abc import Iterable
from typing import Callable, Any, List
import numpy as np
from qiskit.pulse import Schedule, ScheduleBlock, Instruction
from qiskit.pulse.channels import Channel
from qiskit.pulse.schedule import Interval
from qiskit.pulse.exceptions import PulseError
def with_channels(channels: Iterable[Channel] | Channel) -> Callable:
    """Channel filter generator.

    Args:
        channels: List of channels to filter.

    Returns:
        A callback function to filter channels.
    """
    channels = _if_scalar_cast_to_list(channels)

    @singledispatch
    def channel_filter(time_inst):
        """A catch-TypeError function which will only get called if none of the other decorated
        functions, namely handle_numpyndarray() and handle_instruction(), handle the type passed.
        """
        raise TypeError(f"Type '{type(time_inst)}' is not valid data format as an input to channel_filter.")

    @channel_filter.register
    def handle_numpyndarray(time_inst: np.ndarray) -> bool:
        """Filter channel.

        Args:
            time_inst (numpy.ndarray([int, Instruction])): Time

        Returns:
            If instruction matches with condition.
        """
        return any((chan in channels for chan in time_inst[1].channels))

    @channel_filter.register
    def handle_instruction(inst: Instruction) -> bool:
        """Filter channel.

        Args:
            inst: Instruction

        Returns:
            If instruction matches with condition.
        """
        return any((chan in channels for chan in inst.channels))
    return channel_filter