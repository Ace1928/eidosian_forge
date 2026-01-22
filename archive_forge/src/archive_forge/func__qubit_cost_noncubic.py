import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def _qubit_cost_noncubic(n, eta, error, br, charge, vectors):
    """Return the number of logical qubits needed to implement the first quantization
        algorithm for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            int: number of logical qubits needed to implement the first quantization algorithm
        """
    lambda_total, _ = FirstQuantization._norm_noncubic(n, eta, error, br, charge, vectors)
    omega = np.abs(np.sum(np.cross(vectors[0], vectors[1]) * vectors[2]))
    recip_vectors = 2 * np.pi / omega * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
    l_z = eta + charge
    l_nu = 2 * np.pi * n ** (2 / 3)
    n_p = np.ceil(np.log2(n ** (1 / 3) + 1))
    bmin = np.min(np.linalg.svd(recip_vectors)[1])
    alpha = 0.0248759298
    error_t, error_r, error_m, error_b = [alpha * error] * 4
    n_t = int(np.ceil(np.log2(np.pi * lambda_total / error_t)))
    n_r = int(np.ceil(np.log2(eta * l_z * l_nu / (error_r * omega ** (1 / 3)))))
    n_m = int(np.ceil(np.log2(8 * np.pi * eta / (error_m * omega * bmin ** 2) * (eta - 1 + 2 * l_z) * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)))))
    n_errors = 4
    error_qpe = np.sqrt(error ** 2 * (1 - (n_errors * alpha) ** 2))
    clean_temp_H_cost = max([5 * n_r - 4, 5 * n_p + 1]) + max([5, n_m + 3 * n_p])
    reflection_cost = np.ceil(np.log2(eta + 2 * l_z)) + 2 * np.ceil(np.log2(eta)) + 6 * n_p + n_m + 16 + 3
    clean_temp_cost = max([clean_temp_H_cost, reflection_cost])
    clean_cost = 3 * eta * n_p
    clean_cost += np.ceil(np.log2(np.ceil(np.pi * lambda_total / (2 * error_qpe))))
    clean_cost += 1 + 1 + np.ceil(np.log2(eta + 2 * l_z)) + 3 + 3
    clean_cost += 2 * np.ceil(np.log2(eta)) + 5 + 3 * (n_p + 1)
    clean_cost += n_p + n_m + 3 * n_p + 2 + 2 * n_p + 1 + 1 + 2 + 2 * n_p + 6 + 1
    clean_cost += clean_temp_cost
    n_b = np.ceil(np.log2(4 * np.pi * eta * 2 ** (2 * n_p - 2) * np.abs(np.matrix(recip_vectors) @ np.matrix(recip_vectors).T).flatten().sum() / error_b))
    clean_cost += np.max([n_r + 1, n_t, n_b]) + 6 + n_m + 1
    return int(np.ceil(clean_cost))