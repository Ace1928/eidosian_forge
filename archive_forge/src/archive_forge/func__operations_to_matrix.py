import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def _operations_to_matrix(operations, qubits):
    return cirq.Circuit(operations).unitary(qubit_order=cirq.QubitOrder.explicit(qubits), qubits_that_should_be_present=qubits)