import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def _norm_abs_path(file_path):
    return os.path.normpath(os.path.abspath(file_path))