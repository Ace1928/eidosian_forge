import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
def _load_image(path):
    img = Image.open(path)
    if img.mode != 'RGBA' or img.getextrema()[3][0] == 255:
        img = img.convert('RGB')
    return np.asarray(img)