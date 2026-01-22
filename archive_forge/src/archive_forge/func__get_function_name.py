import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def _get_function_name(func):
    filename, _lineno, _name, line = traceback.extract_stack()[-4]
    module, _file_extension = os.path.splitext(filename)
    module = module.replace('/', '.')
    if module.endswith(func.__module__):
        return '%s.[%s].%s' % (func.__module__, line, func.__name__)
    else:
        return '%s.%s' % (func.__module__, func.__name__)