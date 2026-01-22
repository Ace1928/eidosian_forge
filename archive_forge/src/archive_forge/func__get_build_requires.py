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
def _get_build_requires(self, config_settings, requirements):
    sys.argv = [*sys.argv[:1], *self._global_args(config_settings), 'egg_info']
    try:
        with Distribution.patch():
            self.run_setup()
    except SetupRequirementsError as e:
        requirements += e.specifiers
    return requirements