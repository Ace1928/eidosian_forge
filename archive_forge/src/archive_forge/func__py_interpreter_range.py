import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _py_interpreter_range(py_version: PythonVersion) -> Iterator[str]:
    """
    Yields Python versions in descending order.

    After the latest version, the major-only version will be yielded, and then
    all previous versions of that major version.
    """
    if len(py_version) > 1:
        yield f'py{_version_nodot(py_version[:2])}'
    yield f'py{py_version[0]}'
    if len(py_version) > 1:
        for minor in range(py_version[1] - 1, -1, -1):
            yield f'py{_version_nodot((py_version[0], minor))}'