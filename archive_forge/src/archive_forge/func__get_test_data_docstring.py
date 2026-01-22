import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def _get_test_data_docstring(func, value):
    """Returns a docstring based on the following resolution strategy:
    1. Passed value is not a "primitive" and has a docstring, then use it.
    2. In all other cases return None, i.e the test name is used.
    """
    if not _is_primitive(value) and value.__doc__:
        return value.__doc__
    else:
        return None