import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_quiver_arg_sizes():
    X2 = [1, 2]
    X3 = [1, 2, 3]
    with pytest.raises(ValueError, match='X and Y must be the same size, but X.size is 2 and Y.size is 3.'):
        plt.quiver(X2, X3, X2, X2)
    with pytest.raises(ValueError, match='Argument U has a size 3 which does not match 2, the number of arrow positions'):
        plt.quiver(X2, X2, X3, X2)
    with pytest.raises(ValueError, match='Argument V has a size 3 which does not match 2, the number of arrow positions'):
        plt.quiver(X2, X2, X2, X3)
    with pytest.raises(ValueError, match='Argument C has a size 3 which does not match 2, the number of arrow positions'):
        plt.quiver(X2, X2, X2, X2, X3)