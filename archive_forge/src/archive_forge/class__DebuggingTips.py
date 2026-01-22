import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
class _DebuggingTips(SetuptoolsWarning):
    _SUMMARY = 'Problem in editable installation.'
    _DETAILS = '\n    An error happened while installing `{project}` in editable mode.\n\n    The following steps are recommended to help debug this problem:\n\n    - Try to install the project normally, without using the editable mode.\n      Does the error still persist?\n      (If it does, try fixing the problem before attempting the editable mode).\n    - If you are using binary extensions, make sure you have all OS-level\n      dependencies installed (e.g. compilers, toolchains, binary libraries, ...).\n    - Try the latest version of setuptools (maybe the error was already fixed).\n    - If you (or your project dependencies) are using any setuptools extension\n      or customization, make sure they support the editable mode.\n\n    After following the steps above, if the problem still persists and\n    you think this is related to how setuptools handles editable installations,\n    please submit a reproducible example\n    (see https://stackoverflow.com/help/minimal-reproducible-example) to:\n\n        https://github.com/pypa/setuptools/issues\n    '
    _SEE_DOCS = 'userguide/development_mode.html'