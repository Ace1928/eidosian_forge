from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
class AlignFunc(AlignmentKind):
    """Allocate instructions at position specified by callback function.

    The position is specified for each instruction of index ``j`` as a
    fractional coordinate in [0, 1] within the specified duration.

    Instructions played on different channels are also arranged in a sequence.
    This alignment is convenient to create dynamical decoupling sequences such as UDD.

    For example, UDD sequence with 10 pulses can be specified with following function.

    .. code-block:: python

        def udd10_pos(j):
        return np.sin(np.pi*j/(2*10 + 2))**2

    .. note::

        This context cannot be QPY serialized because of the callable. If you use this context,
        your program cannot be saved in QPY format.

    """

    def __init__(self, duration: int | ParameterExpression, func: Callable):
        """Create new equispaced context.

        Args:
            duration: Duration of this context. This should be larger than the schedule duration.
                If the specified duration is shorter than the schedule duration,
                no alignment is performed and the input schedule is just returned.
                This duration can be parametrized.
            func: A function that takes an index of sub-schedule and returns the
                fractional coordinate of of that sub-schedule. The returned value should be
                defined within [0, 1]. The pulse index starts from 1.
        """
        super().__init__(context_params=(duration, func))

    @property
    def is_sequential(self) -> bool:
        return True

    @property
    def duration(self):
        """Return context duration."""
        return self._context_params[0]

    @property
    def func(self):
        """Return context alignment function."""
        return self._context_params[1]

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        instruction_duration_validation(self.duration)
        if self.duration < schedule.duration:
            return schedule
        aligned = Schedule.initialize_from(schedule)
        for ind, (_, child) in enumerate(schedule.children):
            _t_center = self.duration * self.func(ind + 1)
            _t0 = int(_t_center - 0.5 * child.duration)
            if _t0 < 0 or _t0 > self.duration:
                raise PulseError('Invalid schedule position t=%d is specified at index=%d' % (_t0, ind))
            aligned.insert(_t0, child, inplace=True)
        return aligned