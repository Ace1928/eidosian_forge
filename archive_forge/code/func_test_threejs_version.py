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
def test_threejs_version():
    configpath = os.path.join(os.path.abspath(ipyvolume.__path__[0]), '..', 'js', 'package.json')
    with open(configpath) as f:
        config = json.load(f)
    major, minor = ipyvolume._version.__version_threejs__.split('.')
    major_js, minor_js, _patch_js = config['dependencies']['three'][1:].split('.')
    version_msg = 'version in python and js side for three js conflect: %s vs %s' % (ipyvolume._version.__version_threejs__, config['dependencies']['three'])
    assert major == major_js and minor == minor_js, version_msg