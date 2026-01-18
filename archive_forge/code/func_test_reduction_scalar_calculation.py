import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_reduction_scalar_calculation():
    rng = np.random.RandomState(0)
    l = Rotation.random(5, random_state=rng)
    r = Rotation.random(10, random_state=rng)
    p = Rotation.random(7, random_state=rng)
    reduced, left_best, right_best = p.reduce(l, r, return_indices=True)
    scalars = np.zeros((len(l), len(p), len(r)))
    for i, li in enumerate(l):
        for j, pj in enumerate(p):
            for k, rk in enumerate(r):
                scalars[i, j, k] = np.abs((li * pj * rk).as_quat()[3])
    scalars = np.reshape(np.moveaxis(scalars, 1, 0), (scalars.shape[1], -1))
    max_ind = np.argmax(np.reshape(scalars, (len(p), -1)), axis=1)
    left_best_check = max_ind // len(r)
    right_best_check = max_ind % len(r)
    assert (left_best == left_best_check).all()
    assert (right_best == right_best_check).all()
    reduced_check = l[left_best_check] * p * r[right_best_check]
    mag = (reduced.inv() * reduced_check).magnitude()
    assert_array_almost_equal(mag, np.zeros(len(p)))