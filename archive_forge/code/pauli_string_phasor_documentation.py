from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
Creates a PauliStringPhasor on the qubits.