import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
def velocity_field():
    Y, X = np.mgrid[-3:3:100j, -3:3:200j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    return (X, Y, U, V)