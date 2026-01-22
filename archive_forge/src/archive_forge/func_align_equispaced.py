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
def align_equispaced(duration: int | ParameterExpression) -> Generator[None, None, None]:
    """Equispaced alignment pulse scheduling context.

    Pulse instructions within this context are scheduled with the same interval spacing such that
    the total length of the context block is ``duration``.
    If the total free ``duration`` cannot be evenly divided by the number of instructions
    within the context, the modulo is split and then prepended and appended to
    the returned schedule. Delay instructions are automatically inserted in between pulses.

    This context is convenient to write a schedule for periodical dynamic decoupling or
    the Hahn echo sequence.

    Examples:

    .. plot::
       :include-source:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        with pulse.build() as hahn_echo:
            with pulse.align_equispaced(duration=100):
                pulse.play(x90, d0)
                pulse.play(x180, d0)
                pulse.play(x90, d0)

        hahn_echo.draw()

    Args:
        duration: Duration of this context. This should be larger than the schedule duration.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the equispaced context for each channel,
        you should use the context independently for channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignEquispaced(duration=duration))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)