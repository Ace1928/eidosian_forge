from __future__ import annotations
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from .plugin import UnitarySynthesisPlugin
The plugin does not support coupling maps.