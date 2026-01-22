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
def _mutable_insert(self, start_time: int, schedule: 'ScheduleComponent') -> 'Schedule':
    """Mutably insert `schedule` into `self` at `start_time`.

        Args:
            start_time: Time to insert the second schedule.
            schedule: Schedule to mutably insert.
        """
    self._add_timeslots(start_time, schedule)
    self._children.append((start_time, schedule))
    self._parameter_manager.update_parameter_table(schedule)
    return self