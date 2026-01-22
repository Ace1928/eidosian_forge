import pytest
from numpy import (
from numpy.testing import (
class Arrayish:
    """
            A generic object that supports the __array_interface__ and hence
            can in principle be converted to a numeric scalar, but is not
            otherwise recognized as numeric, but also happens to support
            multiplication by floats.

            Data should be an object that implements the buffer interface,
            and contains at least 4 bytes.
            """

    def __init__(self, data):
        self._data = data

    @property
    def __array_interface__(self):
        return {'shape': (), 'typestr': '<i4', 'data': self._data, 'version': 3}

    def __mul__(self, other):
        return self