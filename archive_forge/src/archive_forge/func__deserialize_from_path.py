import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def _deserialize_from_path(ext_to_routine, path, type_name):
    """
    Call deserialization routine with path according to extension.
    ext_to_routine should be a dictionary mapping each supported
    file extension to a corresponding loading function.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Could not parse file '{}' as {}: does not exist or is not a file".format(path, type_name))
    root, ext = os.path.splitext(path)
    if ext in {'.gz', '.bz2', '.xz', '.zip'}:
        ext = os.path.splitext(root)[1]
    if ext in ext_to_routine:
        return ext_to_routine[ext](path)
    raise ValueError("Could not parse file '{}' as {}: no deserialization method for files with '{}' extension. Supported extensions: {}".format(path, type_name, ext, ', '.join(sorted(ext_to_routine))))