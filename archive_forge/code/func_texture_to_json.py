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
def texture_to_json(texture, widget):
    if isinstance(texture, ipywebrtc.HasStream):
        return ipywidgets.widget_serialization['to_json'](texture, widget)
    else:
        return image_to_url(texture, widget)