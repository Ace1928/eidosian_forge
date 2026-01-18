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
def test_volshow_max_shape():
    x, y, z = ipyvolume.examples.xyz(shape=32)
    Im = x * y * z
    v = p3.volshow(Im, max_shape=16, extent=[[0, 32]] * 3)
    assert v.data.shape == (16, 16, 16)
    p3.xlim(0, 16)