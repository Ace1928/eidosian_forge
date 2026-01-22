import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _get_hatch(self, gc, rgbFace):
    """
        Create a new hatch pattern
        """
    if rgbFace is not None:
        rgbFace = tuple(rgbFace)
    edge = gc.get_hatch_color()
    if edge is not None:
        edge = tuple(edge)
    dictkey = (gc.get_hatch(), rgbFace, edge)
    oid = self._hatchd.get(dictkey)
    if oid is None:
        oid = self._make_id('h', dictkey)
        self._hatchd[dictkey] = ((gc.get_hatch_path(), rgbFace, edge), oid)
    else:
        _, oid = oid
    return oid