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
def _clean_conversion_cache():
    baseline_images_size = sum((path.stat().st_size for path in Path(mpl.__file__).parent.glob('**/baseline_images/**/*')))
    max_cache_size = 2 * baseline_images_size
    with cbook._lock_path(_get_cache_path()):
        cache_stat = {path: path.stat() for path in _get_cache_path().glob('*')}
        cache_size = sum((stat.st_size for stat in cache_stat.values()))
        paths_by_atime = sorted(cache_stat, key=lambda path: cache_stat[path].st_atime, reverse=True)
        while cache_size > max_cache_size:
            path = paths_by_atime.pop()
            cache_size -= cache_stat[path].st_size
            path.unlink()