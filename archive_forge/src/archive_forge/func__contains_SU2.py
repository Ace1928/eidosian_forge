import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
def _contains_SU2(op_mat, ops_vecs, tol=1e-08):
    """Checks if a given SU(2) matrix is contained in a list of quaternions for a given tolerance.

    Args:
        op_mat (TensorLike): SU(2) matrix for the operation to be searched
        op_vecs (list(TensorLike)): List of quaternion for the operations that makes the search space.
        tol (float): Tolerance for the match to be considered ``True``.

    Returns:
        Tuple(bool, TensorLike): A bool that shows whether an operation similar to the given operations
        was found, and the quaternion representation of the searched operation.
    """
    node_points = qml.math.array(ops_vecs)
    gate_points = qml.math.array([_quaternion_transform(op_mat)])
    tree = KDTree(node_points)
    dist = tree.query(gate_points, workers=-1)[0][0]
    return (dist < tol, gate_points[0])