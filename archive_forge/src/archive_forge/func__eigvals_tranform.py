from typing import Sequence, Callable
import warnings
from functools import reduce, partial
import scipy
import pennylane as qml
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane import transform
from pennylane.typing import TensorLike
@partial(transform, is_informative=True)
def _eigvals_tranform(tape: qml.tape.QuantumTape, k=1, which='SA') -> (Sequence[qml.tape.QuantumTape], Callable):

    def processing_fn(res):
        [qs] = res
        op_wires = [op.wires for op in qs.operations]
        all_wires = qml.wires.Wires.all_wires(op_wires).tolist()
        unique_wires = qml.wires.Wires.unique_wires(op_wires).tolist()
        if len(all_wires) != len(unique_wires):
            warnings.warn('For multiple operations, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.', UserWarning)
            matrix = qml.matrix(qs, wire_order=qs.wires)
            return qml.math.linalg.eigvals(matrix)
        ev = [eigvals(op, k=k, which=which) for op in qs.operations]
        if len(ev) == 1:
            return ev[0]
        return reduce(qml.math.kron, ev)
    return ([tape], processing_fn)