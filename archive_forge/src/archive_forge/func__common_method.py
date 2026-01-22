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
def _common_method(*classes):
    """A function decorator to attach the function to specified classes as a method.

    .. note:: For developer: A method attached through this decorator may hurt readability
        of the codebase, because the method may not be detected by a code editor.
        Thus, this decorator should be used to a limited extent, i.e. huge helper method.
        By using this decorator wisely, we can reduce code maintenance overhead without
        losing readability of the codebase.
    """

    def decorator(method):

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        for cls in classes:
            setattr(cls, method.__name__, wrapper)
        return method
    return decorator