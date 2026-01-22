import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _normalize_string(string: str) -> str:
    return string.replace('.', '_').replace('-', '_').replace(' ', '_')