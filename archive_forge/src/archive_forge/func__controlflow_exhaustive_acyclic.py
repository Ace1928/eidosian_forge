import itertools
import logging
from math import inf
import numpy as np
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.classical import expr, types
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.circuit import (
from qiskit._accelerate import stochastic_swap as stochastic_swap_rs
from qiskit._accelerate import nlayout
from qiskit.transpiler.passes.layout import disjoint_utils
from .utils import get_swap_map_dag
def _controlflow_exhaustive_acyclic(operation: ControlFlowOp):
    """Return True if the entire control-flow operation represents a block that is guaranteed to be
    entered, and does not cycle back to the initial layout."""
    if isinstance(operation, IfElseOp):
        return len(operation.blocks) == 2
    if isinstance(operation, SwitchCaseOp):
        cases = operation.cases()
        if isinstance(operation.target, expr.Expr):
            type_ = operation.target.type
            if type_.kind is types.Bool:
                max_matches = 2
            elif type_.kind is types.Uint:
                max_matches = 1 << type_.width
            else:
                raise RuntimeError(f"unhandled target type: '{type_}'")
        else:
            max_matches = 2 if isinstance(operation.target, Clbit) else 1 << len(operation.target)
        return CASE_DEFAULT in cases or len(cases) == max_matches
    return False