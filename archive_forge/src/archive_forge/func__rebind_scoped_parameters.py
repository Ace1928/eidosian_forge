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
def _rebind_scoped_parameters(self, expression):
    """If the input is a :class:`.ParameterExpression`, rebind any internal
        :class:`.Parameter`\\ s so that their names match their names in the scope.  Other inputs
        are returned unchanged."""
    if not isinstance(expression, ParameterExpression):
        return expression
    return expression.subs({param: Parameter(self._lookup_variable(param).string) for param in expression.parameters})