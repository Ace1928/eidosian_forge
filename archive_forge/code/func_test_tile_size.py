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
def test_tile_size():
    rows, columns, image_width, image_height = ipyvolume.serialize._compute_tile_size((256, 256, 256))
    assert rows == 16
    assert columns == 16
    assert image_width == 256 * 16
    assert image_height == 256 * 16
    rows, columns, image_width, image_height = ipyvolume.serialize._compute_tile_size((254, 254, 254))
    assert rows == 16
    assert columns == 16
    assert image_width == 256 * 16
    assert image_height == 256 * 16
    ipyvolume.serialize.max_texture_width = 256 * 8
    rows, columns, image_width, image_height = ipyvolume.serialize._compute_tile_size((254, 254, 254))
    assert rows == 32
    assert columns == 8
    assert image_width == 256 * 8
    assert image_height == 256 * 32
    ipyvolume.serialize.min_texture_width = 16 * 8
    rows, columns, image_width, image_height = ipyvolume.serialize._compute_tile_size((16, 16, 16))
    assert rows == 2
    assert columns == 8
    assert image_width == 128
    assert image_height == 128
    ipyvolume.serialize.min_texture_width = 16 * 8
    rows, columns, image_width, image_height = ipyvolume.serialize._compute_tile_size((15, 15, 15))
    assert rows == 2
    assert columns == 8
    assert image_width == 128
    assert image_height == 128