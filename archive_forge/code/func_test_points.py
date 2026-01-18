import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
@pytest.mark.parametrize('a,b', [(0, 1), (0, np.inf), (np.inf, 0), (-np.inf, np.inf), (np.inf, -np.inf)])
def test_points(a, b):
    points = (0, 0.25, 0.5, 0.75, 1.0)
    points += tuple((-x for x in points))
    quadrature_points = 15
    interval_sets = []
    count = 0

    def f(x):
        nonlocal count
        if count % quadrature_points == 0:
            interval_sets.append(set())
        count += 1
        interval_sets[-1].add(float(x))
        return 0.0
    quad_vec(f, a, b, points=points, quadrature='gk15', limit=0)
    for p in interval_sets:
        j = np.searchsorted(sorted(points), tuple(p))
        assert np.all(j == j[0])