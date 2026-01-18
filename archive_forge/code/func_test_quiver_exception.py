from __future__ import absolute_import
import os
import shutil
import json
import contextlib
import numpy as np
import pytest
import ipyvolume
import ipyvolume.pylab as p3
import ipyvolume as ipv
import ipyvolume.examples
import ipyvolume.datasets
import ipyvolume.utils
import ipyvolume.serialize
def test_quiver_exception():
    x, y, z, u, v, w = np.random.random((6, 100))
    with pytest.raises(KeyError):
        p3.quiver(x, y, z, u, v, w, vx=u)