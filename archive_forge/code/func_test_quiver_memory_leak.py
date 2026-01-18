import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
@pytest.mark.skipif(platform.python_implementation() != 'CPython', reason='Requires CPython')
def test_quiver_memory_leak():
    fig, ax = plt.subplots()
    Q = draw_quiver(ax)
    ttX = Q.X
    Q.remove()
    del Q
    assert sys.getrefcount(ttX) == 2