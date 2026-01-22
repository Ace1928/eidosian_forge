import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
class DestructoBox:

    def __init__(self, value, destruct):
        self._val = value
        self._destruct = destruct

    def __add__(self, other):
        tmp = self._val + other._val
        if tmp >= self._destruct:
            raise CustomException
        else:
            self._val = tmp
            return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other):
        tmp = self._val * other._val
        if tmp >= self._destruct:
            raise CustomException
        else:
            self._val = tmp
            return self

    def __rmul__(self, other):
        if other == 0:
            return self
        else:
            return self.__mul__(other)