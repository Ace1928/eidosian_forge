from functools import wraps
import pennylane as qml
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Observable, Operation, Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
from .symbolicop import SymbolicOp
def _single_op_eager(op, update_queue=False):
    if op.has_adjoint:
        adj = op.adjoint()
        if update_queue:
            QueuingManager.remove(op)
            QueuingManager.append(adj)
        return adj
    return Adjoint(op)