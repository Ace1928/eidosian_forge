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
def _replace_timeslots(self, time: int, old: 'ScheduleComponent', new: 'ScheduleComponent'):
    """Replace the timeslots of ``old`` if present with the timeslots of ``new``.

        Args:
            time: The time to remove the timeslots for the ``schedule`` component.
            old: Instruction to replace.
            new: Instruction to replace with.
        """
    self._remove_timeslots(time, old)
    self._add_timeslots(time, new)