import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_quiver_number_of_args():
    X = [1, 2]
    with pytest.raises(TypeError, match='takes from 2 to 5 positional arguments but 1 were given'):
        plt.quiver(X)
    with pytest.raises(TypeError, match='takes from 2 to 5 positional arguments but 6 were given'):
        plt.quiver(X, X, X, X, X, X)