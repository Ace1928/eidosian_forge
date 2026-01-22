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
def _locate_interval_index(intervals: list[Interval], interval: Interval, index: int=0) -> int:
    """Using binary search on start times, find an interval.

    Args:
        intervals: A sorted list of non-overlapping Intervals.
        interval: The interval for which the index into intervals will be found.
        index: A running tally of the index, for recursion. The user should not pass a value.

    Returns:
        The index into intervals that new_interval would be inserted to maintain
        a sorted list of intervals.
    """
    if not intervals or len(intervals) == 1:
        return index
    mid_idx = len(intervals) // 2
    mid = intervals[mid_idx]
    if interval[1] <= mid[0] and interval != mid:
        return _locate_interval_index(intervals[:mid_idx], interval, index=index)
    else:
        return _locate_interval_index(intervals[mid_idx:], interval, index=index + mid_idx)