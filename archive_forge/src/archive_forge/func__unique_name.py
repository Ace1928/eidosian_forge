import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
def _unique_name(self, prefix: str, scope: _Scope) -> str:
    table = scope.symbol_map
    name = basename = _escape_invalid_identifier(prefix)
    while name in table or name in _RESERVED_KEYWORDS:
        name = f'{basename}__generated{next(self._counter)}'
    return name