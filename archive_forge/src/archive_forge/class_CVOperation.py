import abc
import copy
import functools
import itertools
import warnings
from enum import IntEnum
from typing import List
import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, eye, kron
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .utils import pauli_eigs
from .pytrees import register_pytree
class CVOperation(CV, Operation):
    """Base class representing continuous-variable quantum gates.

    CV operations provide a special Heisenberg representation, as well as custom methods
    for differentiation.

    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    @classproperty
    def supports_parameter_shift(self):
        """Returns True iff the CV Operation supports the parameter-shift differentiation method.
        This means that it has ``grad_method='A'`` and
        has overridden the :meth:`~.CV._heisenberg_rep` static method.
        """
        return self.grad_method == 'A' and self.supports_heisenberg

    def heisenberg_pd(self, idx):
        """Partial derivative of the Heisenberg picture transform matrix.

        Computed using grad_recipe.

        Args:
            idx (int): index of the parameter with respect to which the
                partial derivative is computed.
        Returns:
            array[float]: partial derivative
        """
        recipe = self.grad_recipe[idx]
        multiplier = 0.5
        a = 1
        shift = np.pi / 2
        default_param_shift = [[multiplier, a, shift], [-multiplier, a, -shift]]
        param_shift = default_param_shift if recipe is None else recipe
        pd = None
        p = self.parameters
        original_p_idx = p[idx]
        for c, _a, s in param_shift:
            p[idx] = _a * original_p_idx + s
            U = self._heisenberg_rep(p)
            if pd is None:
                pd = c * U
            else:
                pd += c * U
        return pd

    def heisenberg_tr(self, wire_order, inverse=False):
        """Heisenberg picture representation of the linear transformation carried
        out by the gate at current parameter values.

        Given a unitary quantum gate :math:`U`, we may consider its linear
        transformation in the Heisenberg picture, :math:`U^\\dagger(\\cdot) U`.

        If the gate is Gaussian, this linear transformation preserves the polynomial order
        of any observables that are polynomials in :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.
        This also means it maps :math:`\\text{span}(\\mathbf{r})` into itself:

        .. math:: U^\\dagger \\mathbf{r}_i U = \\sum_j \\tilde{U}_{ij} \\mathbf{r}_j

        For Gaussian CV gates, this method returns the transformation matrix for
        the current parameter values of the Operation. The method is not defined
        for non-Gaussian (and non-CV) gates.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
            inverse  (bool): if True, return the inverse transformation instead

        Raises:
            RuntimeError: if the specified operation is not Gaussian or is missing the `_heisenberg_rep` method

        Returns:
            array[float]: :math:`\\tilde{U}`, the Heisenberg picture representation of the linear transformation
        """
        p = [qml.math.toarray(a) for a in self.parameters]
        if inverse:
            try:
                p[0] = np.linalg.inv(p[0])
            except np.linalg.LinAlgError:
                p[0] = -p[0]
        U = self._heisenberg_rep(p)
        if U is None:
            raise RuntimeError(f'{self.name} is not a Gaussian operation, or is missing the _heisenberg_rep method.')
        return self.heisenberg_expand(U, wire_order)