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
class CV:
    """A mixin base class denoting a continuous-variable operation."""

    def heisenberg_expand(self, U, wire_order):
        """Expand the given local Heisenberg-picture array into a full-system one.

        Args:
            U (array[float]): array to expand (expected to be of the dimension ``1+2*self.num_wires``)
            wire_order (Wires): global wire order defining which subspace the operator acts on

        Raises:
            ValueError: if the size of the input matrix is invalid or `num_wires` is incorrect

        Returns:
            array[float]: expanded array, dimension ``1+2*num_wires``
        """
        U_dim = len(U)
        nw = len(self.wires)
        if U.ndim > 2:
            raise ValueError('Only order-1 and order-2 arrays supported.')
        if U_dim != 1 + 2 * nw:
            raise ValueError(f'{self.name}: Heisenberg matrix is the wrong size {U_dim}.')
        if len(wire_order) == 0 or len(self.wires) == len(wire_order):
            return U
        if not wire_order.contains_wires(self.wires):
            raise ValueError(f'{self.name}: Some observable wires {self.wires} do not exist on this device with wires {wire_order}')
        wire_indices = wire_order.indices(self.wires)
        dim = 1 + len(wire_order) * 2

        def loc(w):
            """Returns the slice denoting the location of (x_w, p_w) in the basis."""
            ind = 2 * w + 1
            return slice(ind, ind + 2)
        if U.ndim == 1:
            W = np.zeros(dim)
            W[0] = U[0]
            for k, w in enumerate(wire_indices):
                W[loc(w)] = U[loc(k)]
        elif U.ndim == 2:
            W = np.zeros((dim, dim)) if isinstance(self, Observable) else np.eye(dim)
            W[0, 0] = U[0, 0]
            for k1, w1 in enumerate(wire_indices):
                s1 = loc(k1)
                d1 = loc(w1)
                W[d1, 0] = U[s1, 0]
                W[0, d1] = U[0, s1]
                for k2, w2 in enumerate(wire_indices):
                    W[d1, loc(w2)] = U[s1, loc(k2)]
        return W

    @staticmethod
    def _heisenberg_rep(p):
        """Heisenberg picture representation of the operation.

        * For Gaussian CV gates, this method returns the matrix of the linear
          transformation carried out by the gate for the given parameter values.
          The method is not defined for non-Gaussian gates.

          **The existence of this method is equivalent to setting** ``grad_method = 'A'``.

        * For observables, returns a real vector (first-order observables) or
          symmetric matrix (second-order observables) of expansion coefficients
          of the observable.

        For single-mode Operations we use the basis :math:`\\mathbf{r} = (\\I, \\x, \\p)`.
        For multi-mode Operations we use the basis :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.

        .. note::

            For gates, we assume that the inverse transformation is obtained
            by negating the first parameter.

        Args:
            p (Sequence[float]): parameter values for the transformation

        Returns:
            array[float]: :math:`\\tilde{U}` or :math:`q`
        """
        return None

    @classproperty
    def supports_heisenberg(self):
        """Whether a CV operator defines a Heisenberg representation.

        This indicates that it is Gaussian and does not block the use
        of the parameter-shift differentiation method if found between the differentiated gate
        and an observable.

        Returns:
            boolean
        """
        return CV._heisenberg_rep != self._heisenberg_rep