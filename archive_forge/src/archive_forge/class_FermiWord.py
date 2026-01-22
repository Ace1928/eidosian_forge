import re
from copy import copy
from numbers import Number
from numpy import ndarray
import pennylane as qml
class FermiWord(dict):
    """Immutable dictionary used to represent a Fermi word, a product of fermionic creation and
    annihilation operators, that can be constructed from a standard dictionary.

    The keys of the dictionary are tuples of two integers. The first integer represents the
    position of the creation/annihilation operator in the Fermi word and the second integer
    represents the orbital it acts on. The values of the dictionary are one of ``'+'`` or ``'-'``
    symbols that denote creation and annihilation operators, respectively. The operator
    :math:`a^{\\dagger}_0 a_1` can then be constructed as

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w
    a⁺(0) a(1)
    """
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        self.sorted_dic = dict(sorted(operator.items()))
        indices = [i[0] for i in self.sorted_dic.keys()]
        if indices:
            if list(range(max(indices) + 1)) != indices:
                raise ValueError('The operator indices must belong to the set {0, ..., len(operator)-1}.')
        super().__init__(operator)

    @property
    def wires(self):
        """Return wires in a FermiWord."""
        return set((i[1] for i in self.sorted_dic.keys()))

    def __missing__(self, key):
        """Return empty string for a missing key in FermiWord."""
        return ''

    def update(self, item):
        """Restrict updating FermiWord after instantiation."""
        raise TypeError('FermiWord object does not support assignment')

    def __setitem__(self, key, item):
        """Restrict setting items after instantiation."""
        raise TypeError('FermiWord object does not support assignment')

    def __reduce__(self):
        """Defines how to pickle and unpickle a FermiWord. Otherwise, un-pickling
        would cause __setitem__ to be called, which is forbidden on PauliWord.
        For more information, see: https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        return (FermiWord, (dict(self),))

    def __copy__(self):
        """Copy the FermiWord instance."""
        return FermiWord(dict(self.items()))

    def __deepcopy__(self, memo):
        """Deep copy the FermiWord instance."""
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __hash__(self):
        """Hash value of a FermiWord."""
        return hash(frozenset(self.items()))

    def to_string(self):
        """Return a compact string representation of a FermiWord. Each operator in the word is
        represented by the number of the wire it operates on, and a `+` or `-` to indicate either
        a creation or annihilation operator.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w.to_string()
        a⁺(0) a(1)
        """
        if len(self) == 0:
            return 'I'
        symbol_map = {'+': '⁺', '-': ''}
        string = ' '.join(['a' + symbol_map[j] + '(' + i + ')' for i, j in zip([str(i[1]) for i in self.sorted_dic.keys()], self.sorted_dic.values())])
        return string

    def __str__(self):
        """String representation of a FermiWord."""
        return f'{self.to_string()}'

    def __repr__(self):
        """Terminal representation of a FermiWord"""
        return str(self)

    def __add__(self, other):
        """Add a FermiSentence, FermiWord or constant to a FermiWord. Converts both
        elements into FermiSentences, and uses the FermiSentence __add__
        method"""
        self_fs = FermiSentence({self: 1.0})
        if isinstance(other, FermiSentence):
            return self_fs + other
        if isinstance(other, FermiWord):
            return self_fs + FermiSentence({other: 1.0})
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(f'Arithmetic Fermi operations can only accept an array of length 1, but received {other} of length {len(other)}')
            return self_fs + FermiSentence({FermiWord({}): other})
        raise TypeError(f'Cannot add {type(other)} to a FermiWord.')

    def __radd__(self, other):
        """Add a FermiWord to a constant, i.e. `2 + FermiWord({...})`"""
        if isinstance(other, (Number, ndarray)):
            return self.__add__(other)
        raise TypeError(f'Cannot add a FermiWord to {type(other)}.')

    def __sub__(self, other):
        """Subtract a FermiSentence, FermiWord or constant from a FermiWord. Converts both
        elements into FermiSentences (with negative coefficient for `other`), and
        uses the FermiSentence __add__  method"""
        self_fs = FermiSentence({self: 1.0})
        if isinstance(other, FermiWord):
            return self_fs + FermiSentence({other: -1.0})
        if isinstance(other, FermiSentence):
            other_fs = FermiSentence(dict(zip(other.keys(), [-v for v in other.values()])))
            return self_fs + other_fs
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(f'Arithmetic Fermi operations can only accept an array of length 1, but received {other} of length {len(other)}')
            return self_fs + FermiSentence({FermiWord({}): -1 * other})
        raise TypeError(f'Cannot subtract {type(other)} from a FermiWord.')

    def __rsub__(self, other):
        """Subtract a FermiWord to a constant, i.e. `2 - FermiWord({...})`"""
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(f'Arithmetic Fermi operations can only accept an array of length 1, but received {other} of length {len(other)}')
            self_fs = FermiSentence({self: -1.0})
            other_fs = FermiSentence({FermiWord({}): other})
            return self_fs + other_fs
        raise TypeError(f'Cannot subtract a FermiWord from {type(other)}.')

    def __mul__(self, other):
        """Multiply a FermiWord with another FermiWord, a FermiSentence, or a constant.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w * w
        a⁺(0) a(1) a⁺(0) a(1)
        """
        if isinstance(other, FermiWord):
            if len(self) == 0:
                return copy(other)
            if len(other) == 0:
                return copy(self)
            order_final = [i[0] + len(self) for i in other.sorted_dic.keys()]
            other_wires = [i[1] for i in other.sorted_dic.keys()]
            dict_other = dict(zip([(order_idx, other_wires[i]) for i, order_idx in enumerate(order_final)], other.values()))
            dict_self = dict(zip(self.keys(), self.values()))
            dict_self.update(dict_other)
            return FermiWord(dict_self)
        if isinstance(other, FermiSentence):
            return FermiSentence({self: 1}) * other
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(f'Arithmetic Fermi operations can only accept an array of length 1, but received {other} of length {len(other)}')
            return FermiSentence({self: other})
        raise TypeError(f'Cannot multiply FermiWord by {type(other)}.')

    def __rmul__(self, other):
        """Reverse multiply a FermiWord

        Multiplies a FermiWord "from the left" with an object that can't be modified
        to support __mul__ for FermiWord. Will be defaulted in for example
        ``2 * FermiWord({(0, 0): "+"})``, where the ``__mul__`` operator on an integer
        will fail to multiply with a FermiWord"""
        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(f'Arithmetic Fermi operations can only accept an array of length 1, but received {other} of length {len(other)}')
            return FermiSentence({self: other})
        raise TypeError(f'Cannot multiply FermiWord by {type(other)}.')

    def __pow__(self, value):
        """Exponentiate a Fermi word to an integer power.

        >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
        >>> w**3
        a⁺(0) a(1) a⁺(0) a(1) a⁺(0) a(1)
        """
        if value < 0 or not isinstance(value, int):
            raise ValueError('The exponent must be a positive integer.')
        operator = FermiWord({})
        for _ in range(value):
            operator *= self
        return operator