import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _make_failed_import_test(name, suiteClass):
    message = 'Failed to import test module: %s\n%s' % (name, traceback.format_exc())
    return _make_failed_test(name, ImportError(message), suiteClass, message)