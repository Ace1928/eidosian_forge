import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
def _function_called_str(function_name, args, kwargs):
    """Helper function to output a function call"""
    template_str = '{0}({1}, {2})'
    args_str = repr(args)[1:-1]
    kwargs_str = ', '.join(('%s=%s' % (k, v) for k, v in kwargs.items()))
    return template_str.format(function_name, args_str, kwargs_str)