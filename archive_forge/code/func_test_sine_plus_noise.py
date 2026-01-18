import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
def test_sine_plus_noise():
    np.random.seed(0)
    x = np.sin(np.linspace(0, np.pi * 2.0, 50000)) + np.random.uniform(size=50000) * 0.01
    fig, ax = plt.subplots()
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)
    fig.canvas.draw()
    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = path.cleaned(simplify=True)
    assert simplified.vertices.size == 25240