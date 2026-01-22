from typing import Callable, Sequence
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax, raise_pulse_diff_on_qnode
from .gradient_transform import (
def _one_parameter_generators(op):
    """Compute the effective generators :math:`\\{\\Omega_k\\}` of one-parameter groups that
    reproduce the partial derivatives of a parameterized evolution.
    In particular, compute :math:`U` and :math:`\\partial U / \\partial \\theta_k`
    and recombine them into :math:`\\Omega_k = U^\\dagger \\partial U / \\partial \\theta_k`

    Args:
        op (`~.ParametrizedEvolution`): Parametrized evolution for which to compute the generator

    Returns:
        tuple[tensor_like]: The generators for one-parameter groups that reproduce the
        partial derivatives of the parametrized evolution.
        The ``k``\\ th entry of the returned ``tuple`` has the shape ``(*par_shape, 2**N, 2**N)``
        where ``N`` is the number of qubits the evolution acts on and ``par_shape`` is the
        shape of the ``k``\\ th parameter of the evolution.

    The effective generator can be computed from the derivative of the unitary
    matrix corresponding to the full time evolution of a pulse:

    .. math::

        \\Omega_k = U(\\theta)^\\dagger \\frac{\\partial}{\\partial \\theta_k} U(\\theta)

    Here :math:`U(\\theta)` is the unitary matrix of the time evolution due to the pulse
    and :math:`\\theta` are the variational parameters of the pulse.

    See the documentation of pulse_odegen for more details and a mathematical derivation.
    """

    def _compute_matrix(op_data):
        """Parametrized computation of the matrix for the given pulse ``op``."""
        return op(op_data, t=op.t, **op.odeint_kwargs).matrix()

    def _compute_matrix_split(op_data):
        """Parametrized computation of the matrix for the given pulse ``op``.
        Return the real and imaginary parts separately."""
        mat = _compute_matrix(op_data)
        return (mat.real, mat.imag)
    jac_real, jac_imag = jax.jacobian(_compute_matrix_split)(op.data)
    U_dagger = qml.math.conj(_compute_matrix([qml.math.detach(d) for d in op.data]))
    moveax = partial(qml.math.moveaxis, source=(0, 1), destination=(-2, -1))
    return tuple((moveax(qml.math.tensordot(U_dagger, j_r + 1j * j_i, axes=[[0], [0]])) for j_r, j_i in zip(jac_real, jac_imag)))