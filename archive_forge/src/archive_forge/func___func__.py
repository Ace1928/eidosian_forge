from __future__ import absolute_import, division, print_function
import logging
from functools import wraps, update_wrapper
import types
from warnings import warn
from passlib.utils.compat import PY3
@property
def __func__(self):
    """py3 compatible alias"""
    return self.im_func