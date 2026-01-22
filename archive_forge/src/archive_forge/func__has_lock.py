from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def _has_lock(self) -> bool:
    """
        :return: True if we have a lock and if the lockfile still exists

        :raise AssertionError: If our lock-file does not exist
        """
    return self._owns_lock