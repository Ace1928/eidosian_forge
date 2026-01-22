from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class ShouldDispatch:

    def __array_function__(self, function, types, args, kwargs):
        return (types, args, kwargs)