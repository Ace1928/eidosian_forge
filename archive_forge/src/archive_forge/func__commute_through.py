from copy import copy
import logging
from collections import deque
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
@staticmethod
def _commute_through(blocker, run, front=True):
    """
        Pulls `DAGOpNode`s from the front of `run` (or the back, if `front == False`) until it
        encounters a gate which does not commute with `blocker`.

        Returns a pair of lists whose concatenation is `run`.
        """
    if run == []:
        return ([], [])
    run_clone = deque(run)
    commuted = deque([])
    preindex, commutation_rule = (None, None)
    if isinstance(blocker, DAGOpNode):
        preindex = None
        for i, q in enumerate(blocker.qargs):
            if q == run[0].qargs[0]:
                preindex = i
        commutation_rule = None
        if preindex is not None and isinstance(blocker, DAGOpNode) and (blocker.op.base_class in commutation_table):
            commutation_rule = commutation_table[blocker.op.base_class][preindex]
    if commutation_rule is not None:
        while run_clone:
            next_gate = run_clone[0] if front else run_clone[-1]
            if next_gate.name not in commutation_rule:
                break
            if front:
                run_clone.popleft()
                commuted.append(next_gate)
            else:
                run_clone.pop()
                commuted.appendleft(next_gate)
    if front:
        return (list(commuted), list(run_clone))
    else:
        return (list(run_clone), list(commuted))