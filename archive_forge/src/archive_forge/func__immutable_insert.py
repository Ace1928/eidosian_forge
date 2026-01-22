from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
def _immutable_insert(self, start_time: int, schedule: 'ScheduleComponent', name: str | None=None) -> 'Schedule':
    """Return a new schedule with ``schedule`` inserted into ``self`` at ``start_time``.
        Args:
            start_time: Time to insert the schedule.
            schedule: Schedule to insert.
            name: Name of the new ``Schedule``. Defaults to name of ``self``.
        """
    new_sched = Schedule.initialize_from(self, name)
    new_sched._mutable_insert(0, self)
    new_sched._mutable_insert(start_time, schedule)
    return new_sched