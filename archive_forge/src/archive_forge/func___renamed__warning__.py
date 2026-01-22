import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def __renamed__warning__(msg):
    version = classdict.get('__renamed__version__')
    remove_in = classdict.get('__renamed__remove_in__')
    deprecation_warning("%s  The class '%s' has been renamed to '%s'." % (msg, name, new_class.__name__), version=version, remove_in=remove_in, calling_frame=_find_calling_frame(1))