from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def _update_phase_schedule(n, phase_schedule, swap_plus):
    """
    Given phase_schedule initialized to induce a CZ circuit in SWAP-only network and list of SWAP+ boxes
    Update phase_schedule for each SWAP+ according to Algorithm 2, [2]
    """
    layer_order = list(range(n))[-3::-2] + list(range(n))[-2::-2][::-1]
    order_comp = np.argsort(layer_order[::-1])
    for i in layer_order:
        for j in range(i + 1, n):
            if (i, j) not in swap_plus:
                continue
            phase_schedule[j, j], phase_schedule[i, j] = (phase_schedule[i, j], phase_schedule[j, j])
            for k in range(n):
                if k in (i, j):
                    continue
                if order_comp[min(k, j)] < order_comp[i] and phase_schedule[min(k, j), max(k, j)] % 4 != 0:
                    phase = phase_schedule[min(k, j), max(k, j)]
                    phase_schedule[min(k, j), max(k, j)] = 0
                    for l_s in (i, j, k):
                        phase_schedule[l_s, l_s] = (phase_schedule[l_s, l_s] + phase * 3) % 4
                    for l1, l2 in [(i, j), (i, k), (j, k)]:
                        ls = min(l1, l2)
                        lb = max(l1, l2)
                        phase_schedule[ls, lb] = (phase_schedule[ls, lb] + phase * 3) % 4
    return phase_schedule