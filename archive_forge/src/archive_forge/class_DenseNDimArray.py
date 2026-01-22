import functools
from typing import List
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray, ArrayKind
from sympy.utilities.iterables import flatten
class DenseNDimArray(NDimArray):
    _array: List[Basic]

    def __new__(self, *args, **kwargs):
        return ImmutableDenseNDimArray(*args, **kwargs)

    @property
    def kind(self) -> ArrayKind:
        return ArrayKind._union(self._array)

    def __getitem__(self, index):
        """
        Allows to get items from N-dim array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([0, 1, 2, 3], (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]


        Symbolic index:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(1, 1)`:

        >>> a[i, j].subs({i: 1, j: 1})
        3

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex
        index = self._check_index_for_getitem(index)
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            array = [self._array[self._parse_index(i)] for i in eindices]
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._array[index]

    @classmethod
    def zeros(cls, *shape):
        list_length = functools.reduce(lambda x, y: x * y, shape, S.One)
        return cls._new(([0] * list_length,), shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        """
        from sympy.matrices import Matrix
        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')
        return Matrix(self.shape[0], self.shape[1], self._array)

    def reshape(self, *newshape):
        """
        Returns MutableDenseNDimArray instance with new shape. Elements number
        must be        suitable to new shape. The only argument of method sets
        new shape.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
        >>> a.shape
        (2, 3)
        >>> a
        [[1, 2, 3], [4, 5, 6]]
        >>> b = a.reshape(3, 2)
        >>> b.shape
        (3, 2)
        >>> b
        [[1, 2], [3, 4], [5, 6]]

        """
        new_total_size = functools.reduce(lambda x, y: x * y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Expecting reshape size to %d but got prod(%s) = %d' % (self._loop_size, str(newshape), new_total_size))
        return type(self)(self._array, newshape)