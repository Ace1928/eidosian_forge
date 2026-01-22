import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def _add_inc_data(name, chunksize):
    """
    Generate incremental datasets from basic data sets
    """
    points = DATASETS[name]
    ndim = points.shape[1]
    opts = None
    nmin = ndim + 2
    if name == 'some-points':
        opts = 'QJ Pp'
    elif name == 'pathological-1':
        nmin = 12
    chunks = [points[:nmin]]
    for j in range(nmin, len(points), chunksize):
        chunks.append(points[j:j + chunksize])
    new_name = '%s-chunk-%d' % (name, chunksize)
    assert new_name not in INCREMENTAL_DATASETS
    INCREMENTAL_DATASETS[new_name] = (chunks, opts)