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
def _check_is_iterable_of_str(infos, key):
    if np.iterable(infos):
        for info in infos:
            if not isinstance(info, str):
                raise TypeError(f'Invalid type for {key} metadata. Expected iterable of str, not {type(info)}.')
    else:
        raise TypeError(f'Invalid type for {key} metadata. Expected str or iterable of str, not {type(infos)}.')