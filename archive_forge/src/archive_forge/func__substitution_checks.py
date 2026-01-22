import logging
import math
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
def _substitution_checks(self, dag, old_run, new_circ, basis, qubit):
    """
        Returns `True` when it is recommended to replace `old_run` with `new_circ` over `basis`.
        """
    if new_circ is None:
        return False
    has_cals_p = dag.calibrations is not None and len(dag.calibrations) > 0
    uncalibrated_p = not has_cals_p or any((not dag.has_calibration_for(g) for g in old_run))
    if basis is not None:
        uncalibrated_and_not_basis_p = any((g.name not in basis and (not has_cals_p or not dag.has_calibration_for(g)) for g in old_run))
    else:
        uncalibrated_and_not_basis_p = False
    new_error = 0.0
    old_error = 0.0
    if not uncalibrated_and_not_basis_p:
        new_error = self._error(new_circ, qubit)
        old_error = self._error(old_run, qubit)
    return uncalibrated_and_not_basis_p or (uncalibrated_p and new_error < old_error) or (math.isclose(new_error[0], 0) and (not math.isclose(old_error[0], 0)))