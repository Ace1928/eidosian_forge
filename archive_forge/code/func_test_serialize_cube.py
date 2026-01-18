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
def test_serialize_cube():
    cube = np.zeros((100, 200, 300))
    tiles, _tile_shape, _rows, _columns, _slices = ipv.serialize._cube_to_tiles(cube, 0, 1)
    assert len(tiles.shape) == 3
    f = ipv.serialize.StringIO()
    ipv.serialize.cube_to_png(cube, 0, 1, f)
    assert len(f.getvalue()) > 0