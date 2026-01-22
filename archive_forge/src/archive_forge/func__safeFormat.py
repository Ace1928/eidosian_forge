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
def _safeFormat(formatter: Union[types.FunctionType, Type[str]], o: object) -> str:
    """
    Helper function for L{safe_repr} and L{safe_str}.

    Called when C{repr} or C{str} fail. Returns a string containing info about
    C{o} and the latest exception.

    @param formatter: C{str} or C{repr}.
    @type formatter: C{type}
    @param o: Any object.

    @rtype: C{str}
    @return: A string containing information about C{o} and the raised
        exception.
    """
    io = StringIO()
    traceback.print_exc(file=io)
    className = _determineClassName(o)
    tbValue = io.getvalue()
    return '<{} instance at 0x{:x} with {} error:\n {}>'.format(className, id(o), formatter.__name__, tbValue)