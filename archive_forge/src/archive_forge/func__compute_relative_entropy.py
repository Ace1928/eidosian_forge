import functools
import itertools
from string import ascii_letters as ABC
from autoray import numpy as np
from numpy import float64
import pennylane as qml
from . import single_dispatch  # pylint:disable=unused-import
from .matrix_manipulation import _permute_dense_matrix
from .multi_dispatch import diag, dot, scatter_element_add, einsum, get_interface
from .utils import is_abstract, allclose, cast, convert_like, cast_like
def _compute_relative_entropy(rho, sigma, base=None):
    """
    Compute the quantum relative entropy of density matrix rho with respect to sigma.

    .. math::
        S(\\rho\\,\\|\\,\\sigma)=-\\text{Tr}(\\rho\\log\\sigma)-S(\\rho)=\\text{Tr}(\\rho\\log\\rho)-\\text{Tr}(\\rho\\log\\sigma)
        =\\text{Tr}(\\rho(\\log\\rho-\\log\\sigma))

    where :math:`S` is the von Neumann entropy.
    """
    if base:
        div_base = np.log(base)
    else:
        div_base = 1
    evs_rho, u_rho = qml.math.linalg.eigh(rho)
    evs_sig, u_sig = qml.math.linalg.eigh(sigma)
    evs_rho, evs_sig = (np.real(evs_rho), np.real(evs_sig))
    evs_sig = qml.math.where(evs_sig == 0, 0.0, evs_sig)
    rho_nonzero_mask = qml.math.where(evs_rho == 0.0, False, True)
    ent = qml.math.entr(qml.math.where(rho_nonzero_mask, evs_rho, 1.0))
    rho_batched = len(qml.math.shape(rho)) > 2
    sig_batched = len(qml.math.shape(sigma)) > 2
    indices_rho = 'abc' if rho_batched else 'bc'
    indices_sig = 'abd' if sig_batched else 'bd'
    target = 'acd' if rho_batched or sig_batched else 'cd'
    rel = qml.math.einsum(f'{indices_rho},{indices_sig}->{target}', np.conj(u_rho), u_sig, optimize='greedy')
    rel = np.abs(rel) ** 2
    if sig_batched:
        evs_sig = qml.math.expand_dims(evs_sig, 1)
    rel = qml.math.sum(qml.math.where(rel == 0.0, 0.0, np.log(evs_sig) * rel), -1)
    rel = -qml.math.sum(qml.math.where(rho_nonzero_mask, evs_rho * rel, 0.0), -1)
    return (rel - ent) / div_base