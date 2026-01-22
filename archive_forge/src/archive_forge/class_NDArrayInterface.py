from typing import Tuple
import numpy
import scipy.sparse
from .. import base_matrix_interface as base
class NDArrayInterface(base.BaseMatrixInterface):
    """
    An interface to convert constant values to the numpy ndarray class.
    """
    TARGET_MATRIX = numpy.ndarray

    def const_to_matrix(self, value, convert_scalars: bool=False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        if scipy.sparse.issparse(value):
            result = value.A
        elif isinstance(value, numpy.matrix):
            result = value.A
        elif isinstance(value, list):
            result = numpy.asarray(value).T
        else:
            result = numpy.asarray(value)
        if result.dtype in [numpy.float64] + COMPLEX_TYPES:
            return result
        else:
            return result.astype(numpy.float64)

    def identity(self, size):
        return numpy.eye(size)

    def shape(self, matrix) -> Tuple[int, ...]:
        return tuple((int(d) for d in matrix.shape))

    def size(self, matrix):
        """Returns the number of elements in the matrix.
        """
        return numpy.prod(self.shape(matrix))

    def scalar_value(self, matrix):
        return matrix.item()

    def scalar_matrix(self, value, shape: Tuple[int, ...]):
        return numpy.zeros(shape, dtype='float64') + value

    def reshape(self, matrix, size):
        return numpy.reshape(matrix, size, order='F')