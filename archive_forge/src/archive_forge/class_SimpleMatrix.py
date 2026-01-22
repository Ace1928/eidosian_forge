from .sage_helper import _within_sage
from . import number
from .math_basics import is_Interval
class SimpleMatrix(number.SupportsMultiplicationByNumber):
    """
    A simple matrix class that wraps a list of lists.
    """

    def __init__(self, list_of_lists, ring=None):
        if isinstance(list_of_lists, SimpleMatrix):
            list_of_lists = list_of_lists.data
        if ring is not None:
            self.data = [[ring(e) for e in row] for row in list_of_lists]
        else:
            self.data = list_of_lists
        try:
            self.type = type(self.data[0][0])
            self.shape = (len(list_of_lists), len(list_of_lists[0]))
        except IndexError:
            self.type = type(0)
            self.shape = (0, 0)

    def base_ring(self):
        try:
            return self.data[0][0].parent()
        except IndexError:
            return self.type

    @staticmethod
    def identity(ring, n=0):
        return SimpleMatrix([[1 if i == j else 0 for i in range(n)] for j in range(n)], ring)

    def __iter__(self):
        return self.data.__iter__()

    def __repr__(self):
        str_matrix = [[str(x) for x in row] for row in self.data]
        size = max([max([len(x) for x in row]) for row in str_matrix])
        str_rows = []
        for row in str_matrix:
            str_row = ['% *s' % (size, x) for x in row]
            str_rows.append('[' + ' '.join(str_row) + ']')
        result = '\n'.join(str_rows)
        return result

    def __str__(self):
        str_matrix = [[str(x) for x in row] for row in self.data]
        size = max([max([len(x) for x in row]) for row in str_matrix])
        str_rows = []
        for row in str_matrix:
            str_row = ['% *s' % (size, x) for x in row]
            str_rows.append(' [' + ' '.join(str_row) + ']')
        result = '\n'.join(str_rows)
        result = '[' + '\n'.join(str_rows)[1:] + ']'
        return result

    def __getitem__(self, key):
        if type(key) == tuple:
            i, j = key
            if type(i) == slice or type(j) == slice:
                return SimpleMatrix([row[j] if type(j) == slice else [row[j]] for row in (self.data[i] if type(i) == slice else [self.data[i]])])
            if i < 0 or j < 0:
                raise TypeError("Simple matrices don't have negative indices.")
            return self.data[i][j]
        if type(key) == slice:
            return SimpleMatrix(self.data[key])
        if key < 0:
            raise TypeError("Simple matrices don't have negative indices.")
        return self.data[key]

    def _check_indices(self, key):
        if type(key) != tuple:
            raise TypeError('Can only set an entry, not a row of a simple matrix.')
        i, j = key
        if i < 0 or j < 0:
            raise TypeError("Simple matrices don't have negative indices.")
        return key

    def __setitem__(self, key, value):
        i, j = self._check_indices(key)
        self.data[i][j] = value

    def _noalgebra(self, other):
        raise TypeError('To do matrix algebra, please install numpy or run SnapPy in Sage.')

    def entries(self):
        return [x for row in self.data for x in row]

    def list(self):
        return self.entries()

    def dimensions(self):
        return self.shape

    def __neg__(self):
        return SimpleMatrix([[-x for x in row] for row in self.data])

    def _multiply_by_scalar(self, other):
        return SimpleMatrix([[other * e for e in row] for row in self.data])

    def __mul__(self, other):
        if isinstance(other, SimpleMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError('Cannot multiply matrices with %d columns by matrix with %d rows.' % (self.shape[1], other.shape[0]))
            return SimpleMatrix([[sum((self.data[i][j] * other.data[j][k] for j in range(self.shape[1]))) for k in range(other.shape[1])] for i in range(self.shape[0])])
        if isinstance(other, SimpleVector):
            if self.shape[1] != other.shape[0]:
                raise ValueError('Cannot multiply matrix with %d columns by vector of length %d.' % (self.shape[1], other.shape[0]))
            return SimpleVector([sum((self.data[i][j] * other.data[j] for j in range(self.shape[1]))) for i in range(self.shape[0])])
        raise TypeError('SimpleMatrix only supports multiplication by another SimpleMatrix or SimpleVector. Given type was %r.' % type(other))

    def transpose(self):
        return SimpleMatrix([[self.data[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])])

    def __truediv__(self, other):
        if isinstance(other, number.Number):
            return SimpleMatrix([[d / other for d in row] for row in self.data])
        raise TypeError('SimpleMatrix / SimpleMatrix not supported')
    __div__ = __truediv__

    def det(self):
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        raise TypeError('SimpleMatrix determinant supported only for 2x2')

    def trace(self):
        num_rows, num_cols = self.shape
        if num_rows != num_cols:
            raise ValueError('Trace of non-square %dx%d matrix' % self.shape)
        return sum((self.data[i][i] for i in range(num_rows)))

    def __eq__(self, other):
        return self.data == other.data

    def __add__(self, other):
        if not isinstance(other, SimpleMatrix):
            raise TypeError('SimpleMatrix can only be added to SimpleMatrix.')
        if not self.shape == other.shape:
            raise ValueError('Trying to add a %dx%d matrix to a %dx%d matrix' % (other.shape[0], other.shape[1], self.shape[0], self.shape[1]))
        return SimpleMatrix([[e0 + e1 for e0, e1 in zip(row0, row1)] for row0, row1 in zip(self.data, other.data)])

    def __sub__(self, other):
        if not isinstance(other, SimpleMatrix):
            raise TypeError('SimpleMatrix can only be subtracted from SimpleMatrix.')
        if not self.shape == other.shape:
            raise ValueError('Trying to subtract a %dx%d matrix from a %dx%d matrix' % (other.shape[0], other.shape[1], self.shape[0], self.shape[1]))
        return SimpleMatrix([[e0 - e1 for e0, e1 in zip(row0, row1)] for row0, row1 in zip(self.data, other.data)])
    __inv__ = _noalgebra