from typing import Union, Optional, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression
from ..basis_change import QFT
Get the number of required result qubits.

        Args:
            quadratic: A matrix containing the quadratic coefficients.
            linear: An array containing the linear coefficients.
            offset: A constant offset.

        Returns:
            The number of qubits needed to represent the value of the quadratic form
            in twos complement.
        