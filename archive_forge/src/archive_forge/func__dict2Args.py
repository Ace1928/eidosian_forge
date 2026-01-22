import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def _dict2Args(self, dict):
    args = []
    for key in dict.keys():
        args.append(key)
        args.append(dict[key])
    return args