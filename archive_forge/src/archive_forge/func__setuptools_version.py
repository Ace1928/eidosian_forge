from __future__ import division, absolute_import
import sys
import warnings
from typing import TYPE_CHECKING, Any, TypeVar, Union, Optional, Dict
from ._version import __version__  # noqa: E402
def _setuptools_version():
    return __version__.public()