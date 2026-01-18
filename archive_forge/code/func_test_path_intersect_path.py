import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
@pytest.mark.parametrize('phi', np.concatenate([np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135]) + delta for delta in [-1, 0, 1]]))
def test_path_intersect_path(phi):
    eps_array = [1e-05, 1e-08, 1e-10, 1e-12]
    transform = transforms.Affine2D().rotate(np.deg2rad(phi))
    a = Path([(-2, 0), (2, 0)])
    b = transform.transform_path(a)
    assert a.intersects_path(b) and b.intersects_path(a)
    a = Path([(0, 0), (2, 0)])
    b = transform.transform_path(a)
    assert a.intersects_path(b) and b.intersects_path(a)
    a = transform.transform_path(Path([(0, 1), (0, 3)]))
    b = transform.transform_path(Path([(1, 3), (0, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)
    a = transform.transform_path(Path([(0, 1), (0, 3)]))
    b = transform.transform_path(Path([(0, 5), (0, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)
    assert a.intersects_path(a)
    a = transform.transform_path(Path([(0, 0), (5, 5)]))
    b = transform.transform_path(Path([(1, 1), (3, 3)]))
    assert a.intersects_path(b) and b.intersects_path(a)
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(3, 0), (3, 3)]))
    assert not a.intersects_path(b) and (not b.intersects_path(a))
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(0, 6), (0, 7)]))
    assert not a.intersects_path(b) and (not b.intersects_path(a))
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0 + eps, 1), (0 + eps, 5)]))
        assert not a.intersects_path(b) and (not b.intersects_path(a))
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0, 5 + eps), (0, 7)]))
        assert not a.intersects_path(b) and (not b.intersects_path(a))
    for eps in eps_array:
        a = transform.transform_path(Path([(0, 1), (0, 5)]))
        b = transform.transform_path(Path([(0, 5 - eps), (0, 7)]))
        assert a.intersects_path(b) and b.intersects_path(a)
    a = transform.transform_path(Path([(0, 1), (0, 5)]))
    b = transform.transform_path(Path([(0, 1), (0, 2), (0, 5)]))
    assert a.intersects_path(b) and b.intersects_path(a)
    a = transform.transform_path(Path([(1, -1), (0, -1)]))
    b = transform.transform_path(Path([(0, 1), (0.9, 1)]))
    assert not a.intersects_path(b) and (not b.intersects_path(a))
    a = transform.transform_path(Path([(0.0, -5.0), (1.0, -5.0)]))
    b = transform.transform_path(Path([(1.0, 5.0), (0.0, 5.0)]))
    assert not a.intersects_path(b) and (not b.intersects_path(a))