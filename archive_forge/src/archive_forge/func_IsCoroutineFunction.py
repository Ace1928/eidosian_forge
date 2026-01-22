from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def IsCoroutineFunction(fn):
    try:
        return six.PY34 and asyncio.iscoroutinefunction(fn)
    except:
        return False