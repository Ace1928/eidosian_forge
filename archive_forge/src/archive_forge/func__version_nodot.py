import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _version_nodot(version: PythonVersion) -> str:
    return ''.join(map(str, version))