import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
def _generate_gray_code(num_bits):
    """Generate the gray code for ``num_bits`` bits."""
    if num_bits <= 0:
        raise ValueError('Cannot generate the gray code for less than 1 bit.')
    result = [0]
    for i in range(num_bits):
        result += [x + 2 ** i for x in reversed(result)]
    return [format(x, '0%sb' % num_bits) for x in result]