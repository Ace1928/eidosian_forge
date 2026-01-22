import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
class MyStr(str, np.generic):
    pass