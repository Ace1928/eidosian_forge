from __future__ import annotations
import re
import warnings
from traitlets.log import get_logger
from nbformat import v3 as _v_latest
from nbformat.v3 import (
from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, validate
def writes_py(nb, **kwargs):
    """DEPRECATED: use nbconvert"""
    _warn_format()
    return versions[3].writes_py(nb, **kwargs)