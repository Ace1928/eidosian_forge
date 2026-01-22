import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def _selfOfMethod(methodObject):
    """
    Get the object that a bound method is bound to.

    @param methodObject: a bound method
    @type methodObject: L{types.MethodType}

    @return: the C{self} passed to C{methodObject}
    @rtype: L{object}
    """
    return methodObject.__self__