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
class PauliWord(dict):
    """
    Immutable dictionary used to represent a Pauli Word,
    associating wires with their respective operators.
    Can be constructed from a standard dictionary.

    .. note::

        An empty :class:`~.PauliWord` will be treated as the multiplicative
        identity (i.e identity on all wires). Its matrix is the identity matrix
        (trivially the :math:`1\\times 1` one matrix when no ``wire_order`` is passed to
        ``PauliWord({}).to_mat()``).

    **Examples**

    Initializing a Pauli word:

    >>> w = PauliWord({"a": 'X', 2: 'Y', 3: 'Z'})
    >>> w
    X(a) @ Y(2) @ Z(3)

    When multiplying Pauli words together, we obtain a :class:`~PauliSentence` with the resulting ``PauliWord`` as a key and the corresponding coefficient as its value.

    >>> w1 = PauliWord({0:"X", 1:"Y"})
    >>> w2 = PauliWord({1:"X", 2:"Z"})
    >>> w1 @ w2
    -1j * Z(1) @ Z(2) @ X(0)

    We can multiply scalars to Pauli words or add/subtract them, resulting in a :class:`~PauliSentence` instance.

    >>> 0.5 * w1 - 1.5 * w2 + 2
    0.5 * X(0) @ Y(1)
    + -1.5 * X(1) @ Z(2)
    + 2 * I

    """
    __array_priority__ = 1000

    def __missing__(self, key):
        """If the wire is not in the Pauli word,
        then no operator acts on it, so return the Identity."""
        return I

    def __init__(self, mapping):
        """Strip identities from PauliWord on init!"""
        for wire, op in mapping.copy().items():
            if op == I:
                del mapping[wire]
        super().__init__(mapping)

    @property
    def pauli_rep(self):
        """Trivial pauli_rep"""
        return PauliSentence({self: 1.0})

    def __reduce__(self):
        """Defines how to pickle and unpickle a PauliWord. Otherwise, un-pickling
        would cause __setitem__ to be called, which is forbidden on PauliWord.
        For more information, see: https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        return (PauliWord, (dict(self),))

    def __copy__(self):
        """Copy the PauliWord instance."""
        return PauliWord(dict(self.items()))

    def __deepcopy__(self, memo):
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __setitem__(self, key, item):
        """Restrict setting items after instantiation."""
        raise TypeError('PauliWord object does not support assignment')

    def update(self, __m, **kwargs) -> None:
        """Restrict updating PW after instantiation."""
        raise TypeError('PauliWord object does not support assignment')

    def __hash__(self):
        return hash(frozenset(self.items()))

    def _matmul(self, other):
        """Private matrix multiplication that returns (pauli_word, coeff) tuple for more lightweight processing"""
        base, iterator, swapped = (self, other, False) if len(self) > len(other) else (other, self, True)
        result = copy(dict(base))
        coeff = 1
        for wire, term in iterator.items():
            if wire in base:
                factor, new_op = mul_map[term][base[wire]] if swapped else mul_map[base[wire]][term]
                if new_op == I:
                    del result[wire]
                else:
                    coeff *= factor
                    result[wire] = new_op
            elif term != I:
                result[wire] = term
        return (PauliWord(result), coeff)

    def __matmul__(self, other):
        """Multiply two Pauli words together using the matrix product if wires overlap
        and the tensor product otherwise.

        Empty Pauli words are treated as the Identity operator on all wires.

        Args:
            other (PauliWord): The Pauli word to multiply with

        Returns:
            PauliSentence: coeff * new_word
        """
        if isinstance(other, PauliSentence):
            return PauliSentence({self: 1.0}) @ other
        new_word, coeff = self._matmul(other)
        return PauliSentence({new_word: coeff})

    def __mul__(self, other):
        """Multiply a PauliWord by a scalar

        Args:
            other (Scalar): The scalar to multiply the PauliWord with

        Returns:
            PauliSentence
        """
        if isinstance(other, PauliWord):
            warnings.warn('Matrix/Tensor multiplication using the * operator on PauliWords and PauliSentencesis deprecated, use @ instead. Note also that moving forward the product between twoPauliWords will return a PauliSentence({new_word: ceoff}) instead of a tuple (coeff, new_word).The latter can still be achieved via pw1._matmul(pw2) for lightweight processing', qml.PennyLaneDeprecationWarning)
            return self._matmul(other)
        if isinstance(other, TensorLike):
            if not qml.math.ndim(other) == 0:
                raise ValueError(f'Attempting to multiply a PauliWord with an array of dimension {qml.math.ndim(other)}')
            return PauliSentence({self: other})
        raise TypeError(f'PauliWord can only be multiplied by numerical data. Attempting to multiply by {other} of type {type(other)}')
    __rmul__ = __mul__

    def __add__(self, other):
        """Add PauliWord instances and scalars to PauliWord.
        Returns a PauliSentence."""
        if isinstance(other, PauliWord):
            if other == self:
                return PauliSentence({self: 2.0})
            return PauliSentence({self: 1.0, other: 1.0})
        if isinstance(other, TensorLike):
            IdWord = PauliWord({})
            if IdWord == self:
                return PauliSentence({self: 1.0 + other})
            return PauliSentence({self: 1.0, IdWord: other})
        return NotImplemented
    __radd__ = __add__

    def __iadd__(self, other):
        """Inplace addition"""
        return self + other

    def __sub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return self + -1 * other

    def __rsub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return -1 * self + other

    def __truediv__(self, other):
        """Divide a PauliWord by a scalar"""
        if isinstance(other, TensorLike):
            return self * (1 / other)
        raise TypeError(f'PauliWord can only be divided by numerical data. Attempting to divide by {other} of type {type(other)}')

    def commutes_with(self, other):
        """Fast check if two PauliWords commute with each other"""
        wires = set(self) & set(other)
        if not wires:
            return True
        anticom_count = sum((anticom_map[self[wire]][other[wire]] for wire in wires))
        return anticom_count % 2 == 0

    def _commutator(self, other):
        """comm between two PauliWords, returns tuple (new_word, coeff) for faster arithmetic"""
        if self.commutes_with(other):
            return (PauliWord({}), 0.0)
        new_word, coeff = self._matmul(other)
        return (new_word, 2 * coeff)

    def commutator(self, other):
        """
        Compute commutator between a ``PauliWord`` :math:`P` and other operator :math:`O`

        .. math:: [P, O] = P O - O P

        When the other operator is a :class:`~PauliWord` or :class:`~PauliSentence`,
        this method is faster than computing ``P @ O - O @ P``. It is what is being used
        in :func:`~commutator` when setting ``pauli=True``.

        Args:
            other (Union[Operator, PauliWord, PauliSentence]): Second operator

        Returns:
            ~PauliSentence: The commutator result in form of a :class:`~PauliSentence` instances.

        **Examples**

        You can compute commutators between :class:`~PauliWord` instances.

        >>> pw = PauliWord({0:"X"})
        >>> pw.commutator(PauliWord({0:"Y"}))
        2j * Z(0)

        You can also compute the commutator with other operator types if they have a Pauli representation.

        >>> pw.commutator(qml.Y(0))
        2j * Z(0)
        """
        if isinstance(other, PauliWord):
            new_word, coeff = self._commutator(other)
            if coeff == 0:
                return PauliSentence({})
            return PauliSentence({new_word: coeff})
        if isinstance(other, qml.operation.Operator):
            op_self = PauliSentence({self: 1.0})
            return op_self.commutator(other)
        if isinstance(other, PauliSentence):
            return -1.0 * other.commutator(self)
        raise NotImplementedError(f'Cannot compute natively a commutator between PauliWord and {other} of type {type(other)}')

    def __str__(self):
        """String representation of a PauliWord."""
        if len(self) == 0:
            return 'I'
        return ' @ '.join((f'{op}({w})' for w, op in self.items()))

    def __repr__(self):
        """Terminal representation for PauliWord"""
        return str(self)

    @property
    def wires(self):
        """Track wires in a PauliWord."""
        return Wires(self)

    def to_mat(self, wire_order=None, format='dense', coeff=1.0):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            coeff (float): Coefficient multiplying the resulting matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli word.

        Raises:
            ValueError: Can't get the matrix of an empty PauliWord.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(f"Can't get the matrix for the specified wire order because it does not contain all the Pauli word's wires {self.wires}")
        if len(self) == 0:
            n = len(wire_order) if wire_order is not None else 0
            return np.diag([coeff] * 2 ** n) if format == 'dense' else coeff * sparse.eye(2 ** n, format=format, dtype='complex128')
        if format == 'dense':
            return coeff * reduce(math.kron, (mat_map[self[w]] for w in wire_order))
        return self._to_sparse_mat(wire_order, coeff)

    def _to_sparse_mat(self, wire_order, coeff):
        """Compute the sparse matrix of the Pauli word times a coefficient, given a wire order.
        See pauli_sparse_matrices.md for the technical details of the implementation."""
        matrix_size = 2 ** len(wire_order)
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype='complex128')
        matrix.data = self._get_csr_data(wire_order, coeff)
        matrix.indices = self._get_csr_indices(wire_order)
        matrix.indptr = _cached_arange(matrix_size + 1)
        return matrix

    def _get_csr_data(self, wire_order, coeff):
        """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]
        matrix_size = 2 ** len(wire_order)
        data = np.empty(matrix_size, dtype=np.complex128)
        current_size = 2
        data[:current_size], _ = _cached_sparse_data(full_word[-1])
        data[:current_size] *= coeff
        for s in full_word[-2::-1]:
            if s == 'I':
                data[current_size:2 * current_size] = data[:current_size]
            elif s == 'X':
                data[current_size:2 * current_size] = data[:current_size]
            elif s == 'Y':
                data[current_size:2 * current_size] = 1j * data[:current_size]
                data[:current_size] *= -1j
            elif s == 'Z':
                data[current_size:2 * current_size] = -data[:current_size]
            current_size *= 2
        return data

    def _get_csr_data_2(self, wire_order, coeff):
        """Computes the sparse matrix data of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]
        nwords = len(full_word)
        if nwords < 2:
            return (np.array([1.0]), self._get_csr_data(wire_order, coeff))
        outer = self._get_csr_data(wire_order[:nwords // 2], 1.0)
        inner = self._get_csr_data(wire_order[nwords // 2:], coeff)
        return (outer, inner)

    def _get_csr_indices(self, wire_order):
        """Computes the sparse matrix indices of the Pauli word times a coefficient, given a wire order."""
        full_word = [self[wire] for wire in wire_order]
        matrix_size = 2 ** len(wire_order)
        indices = np.empty(matrix_size, dtype=np.int64)
        current_size = 2
        _, indices[:current_size] = _cached_sparse_data(full_word[-1])
        for s in full_word[-2::-1]:
            if s == 'I':
                indices[current_size:2 * current_size] = indices[:current_size] + current_size
            elif s == 'X':
                indices[current_size:2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == 'Y':
                indices[current_size:2 * current_size] = indices[:current_size]
                indices[:current_size] += current_size
            elif s == 'Z':
                indices[current_size:2 * current_size] = indices[:current_size] + current_size
            current_size *= 2
        return indices

    def operation(self, wire_order=None, get_as_tensor=False):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliWord."""
        if len(self) == 0:
            return Identity(wires=wire_order)
        factors = [_make_operation(op, wire) for wire, op in self.items()]
        if get_as_tensor:
            return factors[0] if len(factors) == 1 else Tensor(*factors)
        pauli_rep = PauliSentence({self: 1})
        return factors[0] if len(factors) == 1 else Prod(*factors, _pauli_rep=pauli_rep)

    def hamiltonian(self, wire_order=None):
        """Return :class:`~pennylane.Hamiltonian` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliWord.")
            return Hamiltonian([1], [Identity(wires=wire_order)])
        obs = [_make_operation(op, wire) for wire, op in self.items()]
        return Hamiltonian([1], [obs[0] if len(obs) == 1 else Tensor(*obs)])

    def map_wires(self, wire_map: dict) -> 'PauliWord':
        """Return a new PauliWord with the wires mapped."""
        return self.__class__({wire_map.get(w, w): op for w, op in self.items()})