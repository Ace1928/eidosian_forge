import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences
def _init_tests():
    LOCAL_FREETYPE_VERSION = '2.6.1'
    from matplotlib import ft2font
    if ft2font.__freetype_version__ != LOCAL_FREETYPE_VERSION or ft2font.__freetype_build_type__ != 'local':
        _log.warning(f'Matplotlib is not built with the correct FreeType version to run tests.  Rebuild without setting system_freetype=1 in mplsetup.cfg.  Expect many image comparison failures below.  Expected freetype version {LOCAL_FREETYPE_VERSION}.  Found freetype version {ft2font.__freetype_version__}.  Freetype build type is {{}}local'.format('' if ft2font.__freetype_build_type__ == 'local' else 'not '))