import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class DummyPd:

    @property
    def dtype(self):
        return PdDtype