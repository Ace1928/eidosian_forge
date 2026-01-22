import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
def _open_setup_script(setup_script):
    if not os.path.exists(setup_script):
        return io.StringIO('from setuptools import setup; setup()')
    return tokenize.open(setup_script)