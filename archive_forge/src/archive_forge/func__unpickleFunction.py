import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def _unpickleFunction(fullyQualifiedName):
    """
    Convert a function name into a function by importing it.

    This is a synonym for L{twisted.python.reflect.namedAny}, but imported
    locally to avoid circular imports, and also to provide a persistent name
    that can be stored (and deprecated) independently of C{namedAny}.

    @param fullyQualifiedName: The fully qualified name of a function.
    @type fullyQualifiedName: native C{str}

    @return: A function object imported from the given location.
    @rtype: L{types.FunctionType}
    """
    from twisted.python.reflect import namedAny
    return namedAny(fullyQualifiedName)