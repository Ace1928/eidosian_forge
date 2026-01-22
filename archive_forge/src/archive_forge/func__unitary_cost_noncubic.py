import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def _unitary_cost_noncubic(n, eta, error, br, charge, vectors):
    """Return the number of Toffoli gates needed to implement the qubitization unitary
        operator for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            int: the number of Toffoli gates needed to implement the qubitization unitary operator
        """
    lambda_total, aa_steps = FirstQuantization._norm_noncubic(n, eta, error, br, charge, vectors)
    omega = np.abs(np.sum(np.cross(vectors[0], vectors[1]) * vectors[2]))
    recip_vectors = 2 * np.pi / omega * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
    n_tof = 500
    alpha = 0.0248759298
    l_z = eta + charge
    l_nu = 2 * np.pi * n ** (2 / 3)
    n_eta = np.ceil(np.log2(eta))
    n_etaz = np.ceil(np.log2(eta + 2 * l_z))
    n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))
    error_t, error_r, error_m = [alpha * error] * 3
    n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))
    n_r = int(np.ceil(np.log2(eta * l_z * l_nu / (error_r * omega ** (1 / 3)))))
    bmin = np.min(np.linalg.svd(recip_vectors)[1])
    n_m = int(np.ceil(np.log2(8 * np.pi * eta / (error_m * omega * bmin ** 2) * (eta - 1 + 2 * l_z) * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)))))
    e_r = FirstQuantization._cost_qrom(l_z)
    cost = 2 * (n_t + 4 * n_etaz + 2 * br - 12) + 14 * n_eta + 8 * br - 36
    cost += (2 * aa_steps + 1) * (3 * n_p ** 2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
    cost += l_z + e_r + 2 * (2 * n_p + 2 * br - 7) + 12 * eta * n_p
    cost += 5 * (n_p - 1) + 2 + 24 * n_p + 6 * n_p * n_r + 18
    cost += n_etaz + 2 * n_eta + 6 * n_p + n_m + 16
    error_b = alpha * error
    n_b = np.ceil(np.log2(2 * np.pi * eta * 2 ** (2 * n_p - 2) * np.abs(np.matrix(recip_vectors) @ np.matrix(recip_vectors).T).flatten().sum() / error_b))
    n_dirty = FirstQuantization._qubit_cost_noncubic(n, eta, error, br, charge, vectors)
    ms_cost = FirstQuantization._momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa=1)[0]
    cost -= 2 * (3 * 2 + 2 * br - 9)
    cost += 2 * (2 * (2 * (2 ** (4 + 1) - 1) + (n_b - 3) * 4 + 2 ** 4 + (n_p - 2)))
    cost += 8
    cost -= (2 * aa_steps + 1) * (3 * n_p ** 2 + 15 * n_p - 7 + 4 * n_m * (n_p + 1))
    cost += (2 * aa_steps + 1) * ms_cost
    return int(np.ceil(cost))