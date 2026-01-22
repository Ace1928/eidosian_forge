from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import (
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
def _decomposer_2q_from_target(self, target, qubits, approximation_degree):
    qubits_tuple = tuple(sorted(qubits))
    reverse_tuple = qubits_tuple[::-1]
    if qubits_tuple in self._decomposer_cache:
        return self._decomposer_cache[qubits_tuple]
    available_2q_basis = {}
    available_2q_props = {}

    def _replace_parameterized_gate(op):
        if isinstance(op, RXXGate) and isinstance(op.params[0], Parameter):
            op = RXXGate(pi / 2)
        elif isinstance(op, RZXGate) and isinstance(op.params[0], Parameter):
            op = RZXGate(pi / 4)
        return op
    try:
        keys = target.operation_names_for_qargs(qubits_tuple)
        for key in keys:
            op = target.operation_from_name(key)
            if not isinstance(op, Gate):
                continue
            available_2q_basis[key] = _replace_parameterized_gate(op)
            available_2q_props[key] = target[key][qubits_tuple]
    except KeyError:
        pass
    try:
        keys = target.operation_names_for_qargs(reverse_tuple)
        for key in keys:
            if key not in available_2q_basis:
                op = target.operation_from_name(key)
                if not isinstance(op, Gate):
                    continue
                available_2q_basis[key] = _replace_parameterized_gate(op)
                available_2q_props[key] = target[key][reverse_tuple]
    except KeyError:
        pass
    if not available_2q_basis:
        raise TranspilerError(f'Target has no gates available on qubits {qubits} to synthesize over.')
    available_1q_basis = _find_matching_euler_bases(target, qubits_tuple[0])
    decomposers = []

    def is_supercontrolled(gate):
        try:
            operator = Operator(gate)
        except QiskitError:
            return False
        kak = TwoQubitWeylDecomposition(operator.data)
        return isclose(kak.a, pi / 4) and isclose(kak.c, 0.0)

    def is_controlled(gate):
        try:
            operator = Operator(gate)
        except QiskitError:
            return False
        kak = TwoQubitWeylDecomposition(operator.data)
        return isclose(kak.b, 0.0) and isclose(kak.c, 0.0)
    supercontrolled_basis = {k: v for k, v in available_2q_basis.items() if is_supercontrolled(v)}
    for basis_1q, basis_2q in product(available_1q_basis, supercontrolled_basis.keys()):
        props = available_2q_props.get(basis_2q)
        if props is None:
            basis_2q_fidelity = 1.0
        else:
            error = getattr(props, 'error', 0.0)
            if error is None:
                error = 0.0
            basis_2q_fidelity = 1 - error
        if approximation_degree is not None:
            basis_2q_fidelity *= approximation_degree
        decomposer = TwoQubitBasisDecomposer(supercontrolled_basis[basis_2q], euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity)
        decomposers.append(decomposer)
    controlled_basis = {k: v for k, v in available_2q_basis.items() if is_controlled(v)}
    basis_2q_fidelity = {}
    embodiments = {}
    pi2_basis = None
    for k, v in controlled_basis.items():
        strength = 2 * TwoQubitWeylDecomposition(Operator(v).data).a
        props = available_2q_props.get(k)
        if props is None:
            basis_2q_fidelity[strength] = 1.0
        else:
            error = getattr(props, 'error', 0.0)
            if error is None:
                error = 0.0
            basis_2q_fidelity[strength] = 1 - error
        embodiment = XXEmbodiments[v.base_class]
        if len(embodiment.parameters) == 1:
            embodiments[strength] = embodiment.assign_parameters([strength])
        else:
            embodiments[strength] = embodiment
        if isclose(strength, pi / 2) and k in supercontrolled_basis:
            pi2_basis = v
    if approximation_degree is not None:
        basis_2q_fidelity = {k: v * approximation_degree for k, v in basis_2q_fidelity.items()}
    if basis_2q_fidelity:
        for basis_1q in available_1q_basis:
            if isinstance(pi2_basis, CXGate) and basis_1q == 'ZSX':
                pi2_decomposer = TwoQubitBasisDecomposer(pi2_basis, euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity, pulse_optimize=True)
                embodiments.update({pi / 2: XXEmbodiments[pi2_decomposer.gate.base_class]})
            else:
                pi2_decomposer = None
            decomposer = XXDecomposer(basis_fidelity=basis_2q_fidelity, euler_basis=basis_1q, embodiments=embodiments, backup_optimizer=pi2_decomposer)
            decomposers.append(decomposer)
    self._decomposer_cache[qubits_tuple] = decomposers
    return decomposers