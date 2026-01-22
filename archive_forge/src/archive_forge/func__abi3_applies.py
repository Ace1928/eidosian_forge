import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _abi3_applies(python_version: PythonVersion) -> bool:
    """
    Determine if the Python version supports abi3.

    PEP 384 was first implemented in Python 3.2.
    """
    return len(python_version) > 1 and tuple(python_version) >= (3, 2)