import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def _norm_noncubic(n, eta, error, br, charge, vectors):
    """Return the 1-norm of a first-quantized Hamiltonian in the plane-wave basis
        for non-cubic systems.

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            vectors (array[float]): lattice vectors

        Returns:
            float: 1-norm of a first-quantized Hamiltonian in the plane-wave basis
        """
    omega = np.abs(np.sum(np.cross(vectors[0], vectors[1]) * vectors[2]))
    recip_vectors = 2 * np.pi / omega * np.array([np.cross(vectors[i], vectors[j]) for i, j in [(1, 2), (2, 0), (0, 1)]])
    bbt = np.matrix(recip_vectors) @ np.matrix(recip_vectors).T
    orthogonal = np.linalg.norm(bbt - np.array([np.max(b ** 2) for b in recip_vectors]) * np.identity(3)) < 1e-06
    l_z = eta + charge
    alpha = 0.0248759298
    p_th = 0.95
    error_uv = alpha * error
    n_p = int(np.ceil(np.log2(n ** (1 / 3) + 1)))
    n0 = n ** (1 / 3)
    bmin = np.min(np.linalg.svd(recip_vectors)[1])
    n_m = int(np.ceil(np.log2(8 * np.pi * eta / (error_uv * omega * bmin ** 2) * (eta - 1 + 2 * l_z) * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p)))))
    lambda_nu = (4 * np.pi * (np.sqrt(3) * n ** (1 / 3) / 2 - 1) + 3 - 3 / n ** (1 / 3) + 3 * integrate.nquad(lambda x, y: 1 / (x ** 2 + y ** 2), [[1, n0], [1, n0]])[0]) / bmin ** 2
    lambda_nu_1 = lambda_nu + 4 / (2 ** n_m * bmin ** 2) * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))
    p_nu = lambda_nu_1 * bmin ** 2 / 2 ** (n_p + 6)
    p_nu_amp = 0
    aa_steps = 0
    n_steps = 30
    for i in reversed(range(n_steps)):
        probability = np.sin((2 * i + 1) * np.arcsin(np.sqrt(p_nu))) ** 2
        if probability > p_th:
            aa_steps = i
            p_nu_amp = probability
    lambda_u = 4 * np.pi * eta * l_z * lambda_nu / omega
    lambda_v = 2 * np.pi * eta * (eta - 1) * lambda_nu / omega
    lambda_u_1 = lambda_u * lambda_nu_1 / lambda_nu
    lambda_v_1 = lambda_v * lambda_nu_1 / lambda_nu
    b_mat = np.matrix(recip_vectors)
    abs_sum = np.abs(b_mat @ b_mat.T).flatten().sum()
    if orthogonal:
        lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 4
    else:
        lambda_t_p = abs_sum * eta * 2 ** (2 * n_p - 2) / 2
    p_eq = FirstQuantization.success_prob(3 * eta + 2 * charge, br) * FirstQuantization.success_prob(eta, br) ** 2
    if p_nu * lambda_t_p >= (1 - p_nu) * (lambda_u_1 + lambda_v_1):
        raise ValueError('The computed 1-norm is zero.')
    if p_nu_amp * lambda_t_p >= (1 - p_nu_amp) * (lambda_u_1 + lambda_v_1):
        return ((lambda_t_p + lambda_u_1 + lambda_v_1) / p_eq, aa_steps)
    return ((lambda_u_1 + lambda_v_1 / (1 - 1 / eta)) / p_nu_amp / p_eq, aa_steps)