from __future__ import annotations
from typing import Union, Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.synthesis.evolution import EvolutionSynthesis, LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp
def _get_default_label(operator):
    if isinstance(operator, list):
        label = f'exp(-it ({[' + '.join(op.paulis.to_labels()) for op in operator]}))'
    elif len(operator.paulis) == 1:
        label = f'exp(-it {operator.paulis.to_labels()[0]})'
    else:
        label = f'exp(-it ({' + '.join(operator.paulis.to_labels())}))'
    return label