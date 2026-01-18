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
def test_quick():
    x, y, z = ipyvolume.examples.xyz()
    p3.volshow(x * y * z)
    ipyvolume.quickvolshow(x * y * z, lighting=True)
    ipyvolume.quickvolshow(x * y * z, lighting=True, level=1, opacity=1, level_width=1)
    x, y, z, u, v, w = np.random.random((6, 100))
    ipyvolume.quickscatter(x, y, z)
    ipyvolume.quickquiver(x, y, z, u, v, w)