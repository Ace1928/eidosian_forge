import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences
def _add_data_doc(docstring, replace_names):
    """
    Add documentation for a *data* field to the given docstring.

    Parameters
    ----------
    docstring : str
        The input docstring.
    replace_names : list of str or None
        The list of parameter names which arguments should be replaced by
        ``data[name]`` (if ``data[name]`` does not throw an exception).  If
        None, replacement is attempted for all arguments.

    Returns
    -------
    str
        The augmented docstring.
    """
    if docstring is None or (replace_names is not None and len(replace_names) == 0):
        return docstring
    docstring = inspect.cleandoc(docstring)
    data_doc = '    If given, all parameters also accept a string ``s``, which is\n    interpreted as ``data[s]`` (unless this raises an exception).' if replace_names is None else f'    If given, the following parameters also accept a string ``s``, which is\n    interpreted as ``data[s]`` (unless this raises an exception):\n\n    {', '.join(map('*{}*'.format, replace_names))}'
    if _log.level <= logging.DEBUG:
        if 'data : indexable object, optional' not in docstring:
            _log.debug('data parameter docstring error: no data parameter')
        if 'DATA_PARAMETER_PLACEHOLDER' not in docstring:
            _log.debug('data parameter docstring error: missing placeholder')
    return docstring.replace('    DATA_PARAMETER_PLACEHOLDER', data_doc)