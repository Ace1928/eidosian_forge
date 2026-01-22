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
class CVObservable(CV, Observable):
    """Base class representing continuous-variable observables.

    CV observables provide a special Heisenberg representation.

    The class attribute :attr:`~.ev_order` can be defined to indicate
    to PennyLane whether the corresponding CV observable is a polynomial in the
    quadrature operators. If so,

    * ``ev_order = 1`` indicates a first order polynomial in quadrature
      operators :math:`(\\x, \\p)`.

    * ``ev_order = 2`` indicates a second order polynomial in quadrature
      operators :math:`(\\x, \\p)`.

    If :attr:`~.ev_order` is not ``None``, then the Heisenberg representation
    of the observable should be defined in the static method :meth:`~.CV._heisenberg_rep`,
    returning an array of the correct dimension.

    Args:
       params (tuple[tensor_like]): trainable parameters
       wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
           If not given, args[-1] is interpreted as wires.
       id (str): custom label given to an operator instance,
           can be useful for some applications where the instance has to be identified
    """
    ev_order = None

    def heisenberg_obs(self, wire_order):
        """Representation of the observable in the position/momentum operator basis.

        Returns the expansion :math:`q` of the observable, :math:`Q`, in the
        basis :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.

        * For first-order observables returns a real vector such
          that :math:`Q = \\sum_i q_i \\mathbf{r}_i`.

        * For second-order observables returns a real symmetric matrix
          such that :math:`Q = \\sum_{ij} q_{ij} \\mathbf{r}_i \\mathbf{r}_j`.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
        Returns:
            array[float]: :math:`q`
        """
        p = self.parameters
        U = self._heisenberg_rep(p)
        return self.heisenberg_expand(U, wire_order)