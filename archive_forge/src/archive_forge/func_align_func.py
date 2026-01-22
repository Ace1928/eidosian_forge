from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
@contextmanager
def align_func(duration: int | ParameterExpression, func: Callable[[int], float]) -> Generator[None, None, None]:
    """Callback defined alignment pulse scheduling context.

    Pulse instructions within this context are scheduled at the location specified by
    arbitrary callback function `position` that takes integer index and returns
    the associated fractional location within [0, 1].
    Delay instruction is automatically inserted in between pulses.

    This context may be convenient to write a schedule of arbitrary dynamical decoupling
    sequences such as Uhrig dynamical decoupling.

    Examples:

    .. plot::
       :include-source:

        import numpy as np
        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        def udd10_pos(j):
            return np.sin(np.pi*j/(2*10 + 2))**2

        with pulse.build() as udd_sched:
            pulse.play(x90, d0)
            with pulse.align_func(duration=300, func=udd10_pos):
                for _ in range(10):
                    pulse.play(x180, d0)
            pulse.play(x90, d0)

        udd_sched.draw()

    Args:
        duration: Duration of context. This should be larger than the schedule duration.
        func: A function that takes an index of sub-schedule and returns the
            fractional coordinate of of that sub-schedule.
            The returned value should be defined within [0, 1].
            The pulse index starts from 1.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the numerical context for each channel,
        you need to apply the context independently to channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignFunc(duration=duration, func=func))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)