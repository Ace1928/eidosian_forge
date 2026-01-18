import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
def to_mat(self, wire_order=None, format='dense', buffer_size=None):
    """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            buffer_size (int or None): The maximum allowed memory in bytes to store intermediate results
                in the calculation of sparse matrices. It defaults to ``2 ** 30`` bytes that make
                1GB of memory. In general, larger buffers allow faster computations.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli sentence.

        Raises:
            ValueError: Can't get the matrix of an empty PauliSentence.
        """
    wire_order = self.wires if wire_order is None else Wires(wire_order)
    if not wire_order.contains_wires(self.wires):
        raise ValueError(f"Can't get the matrix for the specified wire order because it does not contain all the Pauli sentence's wires {self.wires}")

    def _pw_wires(w: Iterable) -> Wires:
        """Return the native Wires instance for a list of wire labels.
            w represents the wires of the PauliWord being processed. In case
            the PauliWord is empty ({}), choose any arbitrary wire from the
            PauliSentence it is composed in.
            """
        return w or Wires(self.wires[0]) if self.wires else self.wires
    if len(self) == 0:
        n = len(wire_order) if wire_order is not None else 0
        if format == 'dense':
            return np.zeros((2 ** n, 2 ** n))
        return sparse.csr_matrix((2 ** n, 2 ** n), dtype='complex128')
    if format != 'dense':
        return self._to_sparse_mat(wire_order, buffer_size=buffer_size)
    mats_and_wires_gen = ((coeff * pw.to_mat(wire_order=_pw_wires(pw.wires), format=format), _pw_wires(pw.wires)) for pw, coeff in self.items())
    reduced_mat, result_wire_order = math.reduce_matrices(mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add)
    return math.expand_matrix(reduced_mat, result_wire_order, wire_order=wire_order)