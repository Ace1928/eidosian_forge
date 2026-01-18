import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
@pytest.mark.skipif(platform.python_implementation() != 'CPython', reason='Requires CPython')
def test_quiver_key_memory_leak():
    fig, ax = plt.subplots()
    Q = draw_quiver(ax)
    qk = ax.quiverkey(Q, 0.5, 0.92, 2, '$2 \\frac{m}{s}$', labelpos='W', fontproperties={'weight': 'bold'})
    assert sys.getrefcount(qk) == 3
    qk.remove()
    assert sys.getrefcount(qk) == 2