from __future__ import absolute_import
import logging
import warnings
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import ipywebrtc
import pythreejs
import traitlets
from traitlets import Unicode, Integer
from traittypes import Array
from bqplot import scales
import ipyvolume
import ipyvolume as ipv  # we should not have ipyvolume under two names either
import ipyvolume._version
from ipyvolume.traittypes import Image
from ipyvolume.serialize import (
from ipyvolume.transferfunction import TransferFunction
from ipyvolume.utils import debounced, grid_slice, reduce_size
@traitlets.default('camera')
def _default_camera(self):
    z = 2 * np.tan(45.0 / 2.0 * np.pi / 180) / np.tan(self.camera_fov / 2.0 * np.pi / 180)
    return pythreejs.PerspectiveCamera(fov=self.camera_fov, position=(0, 0, z), aspect=1)