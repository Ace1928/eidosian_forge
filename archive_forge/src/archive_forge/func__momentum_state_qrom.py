import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def _momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa):
    """Returns the Toffoli cost for preparing the momentum state superposition.

        Derived from Section D.1 item (6) and Appendix K.1.f of arXiv:2302.07981v1 (2023)"""
    x = 2 ** (3 * n_p)
    beta_dirty = max([np.floor(n_dirty / n_m), 1])
    beta_parallel = max([np.floor(n_tof / kappa), 1])
    if n_tof == 1:
        beta_gate = max([np.floor(np.sqrt(2 * x / (3 * n_m))), 1])
        beta = np.min([beta_dirty, beta_gate])
        ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * n_m * beta
    else:
        beta_gate = max([np.floor(2 * x / (3 * n_m / kappa) * np.log(2)), 1])
        beta = np.min([beta_dirty, beta_gate, beta_parallel])
        ms_cost_qrom = 2 * np.ceil(x / beta) + 3 * np.ceil(n_m / kappa) * np.ceil(np.log2(beta))
    ms_cost = 2 * ms_cost_qrom + n_m + 8 * (n_p - 1) + 6 * n_p + 2 + 2 * n_p + n_m + 2
    return (ms_cost, beta)