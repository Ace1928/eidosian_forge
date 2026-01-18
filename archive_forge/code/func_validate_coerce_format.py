import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
def validate_coerce_format(fmt):
    """
    Validate / coerce a user specified image format, and raise an informative
    exception if format is invalid.

    Parameters
    ----------
    fmt
        A value that may or may not be a valid image format string.

    Returns
    -------
    str or None
        A valid image format string as supported by orca. This may not
        be identical to the input image designation. For example,
        the resulting string will always be lower case and  'jpg' is
        converted to 'jpeg'.

        If the input format value is None, then no exception is raised and
        None is returned.

    Raises
    ------
    ValueError
        if the input `fmt` cannot be interpreted as a valid image format.
    """
    if fmt is None:
        return None
    if not isinstance(fmt, str) or not fmt:
        raise_format_value_error(fmt)
    fmt = fmt.lower()
    if fmt[0] == '.':
        fmt = fmt[1:]
    if fmt not in format_conversions:
        raise_format_value_error(fmt)
    return format_conversions[fmt]