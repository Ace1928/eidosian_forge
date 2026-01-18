from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def volume_to_json_volume_tiled(vol, obj=None):
    if vol is None:
        return None
    vol = np.asarray(vol)
    vol_shape = vol.shape[-3:][::-1]
    a = math.sqrt(float(vol_shape[2]) / float(vol_shape[0] * vol_shape[1]))
    tile_shape = [int(math.ceil(vol_shape[1] * a)), int(math.ceil(vol_shape[0] * a))]
    tex_size = [vol_shape[1] * tile_shape[1], vol_shape[0] * tile_shape[0]]
    if vol.ndim == 4:
        return {'volume_data_tiled': [tile_volume(vol[t], tex_size, tile_shape, vol_shape) for t in range(vol.shape[0])], 'shape': vol_shape, 'tile_shape': tile_shape, 'vol_tex_size': tex_size}
    else:
        return {'volume_data_tiled': [tile_volume(vol, tex_size, tile_shape, vol_shape)], 'shape': vol_shape, 'tile_shape': tile_shape, 'vol_tex_size': tex_size}
    return None