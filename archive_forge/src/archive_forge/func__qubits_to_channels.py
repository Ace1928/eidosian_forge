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
def _qubits_to_channels(*channels_or_qubits: int | chans.Channel) -> set[chans.Channel]:
    """Returns the unique channels of the input qubits."""
    channels = set()
    for channel_or_qubit in channels_or_qubits:
        if isinstance(channel_or_qubit, int):
            channels |= qubit_channels(channel_or_qubit)
        elif isinstance(channel_or_qubit, chans.Channel):
            channels.add(channel_or_qubit)
        else:
            raise exceptions.PulseError(f'{channel_or_qubit} is not a "Channel" or qubit (integer).')
    return channels