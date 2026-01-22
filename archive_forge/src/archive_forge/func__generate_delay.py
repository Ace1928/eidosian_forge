import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
def _generate_delay(time: float):
    if int(time) != time:
        raise QASM2ParseError("the custom 'delay' instruction can only accept an integer parameter")
    return Delay(int(time), unit='dt')