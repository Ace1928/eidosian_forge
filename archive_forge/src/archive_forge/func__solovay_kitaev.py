import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
def _solovay_kitaev(umat, n, u_n1_ids, u_n1_mat):
    """Recursive method as given in the Section 3 of arXiv:0505030"""
    if not n:
        seq_node = qml.math.array([_quaternion_transform(umat)])
        _, [index] = kd_tree.query(seq_node, workers=-1)
        return (approx_set_ids[index], approx_set_mat[index])
    v_n, w_n = _group_commutator_decompose(umat @ qml.math.conj(qml.math.transpose(u_n1_mat)))
    c_ids_mats = []
    for c_n in [v_n, w_n]:
        c_n1_ids, c_n1_mat = (None, None)
        for i in range(n):
            c_n1_ids, c_n1_mat = _solovay_kitaev(c_n, i, c_n1_ids, c_n1_mat)
        c_n1_ids_adj = [qml.adjoint(gate, lazy=False) for gate in reversed(c_n1_ids)]
        c_n1_mat_adj = qml.math.conj(qml.math.transpose(c_n1_mat))
        c_ids_mats.append([c_n1_ids, c_n1_mat, c_n1_ids_adj, c_n1_mat_adj])
    v_n1_ids, v_n1_mat, v_n1_ids_adj, v_n1_mat_adj = c_ids_mats[0]
    w_n1_ids, w_n1_mat, w_n1_ids_adj, w_n1_mat_adj = c_ids_mats[1]
    approx_ids = u_n1_ids + w_n1_ids_adj + v_n1_ids_adj + w_n1_ids + v_n1_ids
    approx_mat = v_n1_mat @ w_n1_mat @ v_n1_mat_adj @ w_n1_mat_adj @ u_n1_mat
    return (approx_ids, approx_mat)