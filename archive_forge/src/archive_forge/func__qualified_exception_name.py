from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _qualified_exception_name(eclass, unqualified_breezy_errors=False):
    """Give name of error class including module for non-builtin exceptions

    If `unqualified_breezy_errors` is True, errors specific to breezy will
    also omit the module prefix.
    """
    class_name = eclass.__name__
    module_name = eclass.__module__
    if module_name in ('builtins', 'exceptions', '__main__') or (unqualified_breezy_errors and module_name == 'breezy.errors'):
        return class_name
    return '{}.{}'.format(module_name, class_name)