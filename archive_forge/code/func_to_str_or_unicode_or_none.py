import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@staticmethod
def to_str_or_unicode_or_none(v):
    """
        Convert a value to a string if it's not None, a string,
        or a unicode (on Python 2).
        """
    if v is None or isinstance(v, str):
        return v
    else:
        return str(v)