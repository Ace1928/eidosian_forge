from __future__ import annotations
from typing import Union, Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.synthesis.evolution import EvolutionSynthesis, LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp
Gate parameters should be int, float, or ParameterExpression