import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
def _switch_case_eq(node1, node2, bit_indices1, bit_indices2):
    target1 = node1.op.target
    target2 = node2.op.target
    if isinstance(target1, expr.Expr) and isinstance(target2, expr.Expr):
        if not expr.structurally_equivalent(target1, target2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)):
            return False
    elif isinstance(target1, Clbit) and isinstance(target2, Clbit):
        if bit_indices1[target1] != bit_indices2[target2]:
            return False
    elif isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        if target1.size != target2.size or any((bit_indices1[b1] != bit_indices2[b2] for b1, b2 in zip(target1, target2))):
            return False
    else:
        return False
    cases1 = [case for case, _ in node1.op.cases_specifier()]
    cases2 = [case for case, _ in node2.op.cases_specifier()]
    return len(cases1) == len(cases2) and all((set(labels1) == set(labels2) for labels1, labels2 in zip(cases1, cases2))) and (len(node1.op.blocks) == len(node2.op.blocks)) and all((_circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1) == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2) for block1, block2 in zip(node1.op.blocks, node2.op.blocks)))