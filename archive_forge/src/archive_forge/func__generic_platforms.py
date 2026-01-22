import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())