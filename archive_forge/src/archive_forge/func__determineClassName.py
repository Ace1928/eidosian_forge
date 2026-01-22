import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
def _determineClassName(x):
    c = _determineClass(x)
    try:
        return c.__name__
    except BaseException:
        try:
            return str(c)
        except BaseException:
            return '<BROKEN CLASS AT 0x%x>' % id(c)