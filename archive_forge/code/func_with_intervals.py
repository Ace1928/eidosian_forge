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
def with_intervals(ranges: Iterable[Interval] | Interval) -> Callable:
    """Interval filter generator.

    Args:
        ranges: List of intervals ``[t0, t1]`` to filter.

    Returns:
        A callback function to filter intervals.
    """
    ranges = _if_scalar_cast_to_list(ranges)

    def interval_filter(time_inst) -> bool:
        """Filter interval.
        Args:
            time_inst (Tuple[int, Instruction]): Time

        Returns:
            If instruction matches with condition.
        """
        for t0, t1 in ranges:
            inst_start = time_inst[0]
            inst_stop = inst_start + time_inst[1].duration
            if t0 <= inst_start and inst_stop <= t1:
                return True
        return False
    return interval_filter