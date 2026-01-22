import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class Foldername(Path):
    """
    Parameter that can be set to a string specifying the path of a folder.

    The string should be specified in UNIX style, but it will be
    returned in the format of the user's operating system.

    The specified path can be absolute, or relative to either:

    * any of the paths specified in the search_paths attribute (if
      search_paths is not None);

    or

    * any of the paths searched by resolve_dir_path() (if search_paths
      is None).
    """

    def _resolve(self, path):
        return resolve_path(path, path_to_file=False, search_paths=self.search_paths)