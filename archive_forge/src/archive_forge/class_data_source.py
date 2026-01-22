import sys
import pytest
import numpy as np
from numpy.testing import extbuild
class data_source:
    """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

    def __init__(self, size, value):
        self.size = size
        self.value = value

    @property
    def __array_struct__(self):
        return get_module.new_array_struct(self.size, self.value)