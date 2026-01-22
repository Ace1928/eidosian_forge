from copy import copy
from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary
class SparseHamiltonian(Observable):
    """
    A Hamiltonian represented directly as a sparse matrix in Compressed Sparse Row (CSR) format.

    .. warning::

        ``SparseHamiltonian`` observables can only be used to return expectation values.
        Variances and samples are not supported.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        H (csr_matrix): a sparse matrix in SciPy Compressed Sparse Row (CSR) format with
            dimension :math:`(2^n, 2^n)`, where :math:`n` is the number of wires.
        wires (Sequence[int]): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    Sparse Hamiltonians can be constructed directly with a SciPy-compatible sparse matrix.

    Alternatively, you can construct your Hamiltonian as usual using :class:`~.Hamiltonian`, and then use
    :meth:`~.Hamiltonian.sparse_matrix` to construct the sparse matrix that serves as the input
    to ``SparseHamiltonian``:

    >>> wires = range(20)
    >>> coeffs = [1 for _ in wires]
    >>> observables = [qml.Z(i) for i in wires]
    >>> H = qml.Hamiltonian(coeffs, observables)
    >>> Hmat = H.sparse_matrix()
    >>> H_sparse = qml.SparseHamiltonian(Hmat, wires)
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    grad_method = None

    def __init__(self, H, wires=None, id=None):
        if not isinstance(H, csr_matrix):
            raise TypeError('Observable must be a scipy sparse csr_matrix.')
        super().__init__(H, wires=wires, id=id)
        self.H = H
        mat_len = 2 ** len(self.wires)
        if H.shape != (mat_len, mat_len):
            raise ValueError(f'Sparse Matrix must be of shape ({mat_len}, {mat_len}). Got {H.shape}.')

    def __mul__(self, value):
        """The scalar multiplication operation between a scalar and a SparseHamiltonian."""
        if not isinstance(value, (int, float)) and qml.math.ndim(value) != 0:
            raise TypeError(f'Scalar value must be an int or float. Got {type(value)}')
        return qml.SparseHamiltonian(csr_matrix.multiply(self.H, value), wires=self.wires)
    __rmul__ = __mul__

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or '𝓗', cache=cache)

    @staticmethod
    def compute_matrix(H):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SparseHamiltonian.matrix`


        This method returns a dense matrix. For a sparse matrix representation, see
        :meth:`~.SparseHamiltonian.compute_sparse_matrix`.

        Args:
            H (scipy.sparse.csr_matrix): sparse matrix used to create the operator

        Returns:
            array: dense matrix

        **Example**

        >>> from scipy.sparse import csr_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = csr_matrix(H)
        >>> res = qml.SparseHamiltonian.compute_matrix(H)
        >>> res
        [[ 6.+0.j  1.-2.j]
         [ 1.+2.j -1.+0.j]]
        >>> type(res)
        <class 'numpy.ndarray'>
        """
        return H.toarray()

    @staticmethod
    def compute_sparse_matrix(H):
        """Representation of the operator as a sparse canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SparseHamiltonian.sparse_matrix`

        This method returns a sparse matrix. For a dense matrix representation, see
        :meth:`~.SparseHamiltonian.compute_matrix`.

        Args:
            H (scipy.sparse.csr_matrix): sparse matrix used to create the operator

        Returns:
            scipy.sparse.csr_matrix: sparse matrix

        **Example**

        >>> from scipy.sparse import csr_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = csr_matrix(H)
        >>> res = qml.SparseHamiltonian.compute_sparse_matrix(H)
        >>> res
        (0, 0)	(6+0j)
        (0, 1)	(1-2j)
        (1, 0)	(1+2j)
        (1, 1)	(-1+0j)
        >>> type(res)
        <class 'scipy.sparse.csr_matrix'>
        """
        return H