import contextlib
import functools
import inspect
import os
from platform import uname
from pathlib import Path
import shutil
import string
import sys
import warnings
from packaging.version import parse as parse_version
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure
def _raise_on_image_difference(expected, actual, tol):
    __tracebackhide__ = True
    err = compare_images(expected, actual, tol, in_decorator=True)
    if err:
        for key in ['actual', 'expected', 'diff']:
            err[key] = os.path.relpath(err[key])
        raise ImageComparisonFailure('images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s\n\t%(diff)s' % err)