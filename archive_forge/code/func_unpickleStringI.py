import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def unpickleStringI(val, sek):
    """
    Convert the output of L{pickleStringI} into an appropriate type for the
    current Python version.

    This may be called on Python 3 and will convert a cStringIO into an
    L{io.StringIO}.

    @param val: The content of the file.
    @type val: L{bytes}

    @param sek: The seek position of the file.
    @type sek: L{int}

    @return: a file-like object which you can read bytes from.
    @rtype: C{cStringIO.OutputType} on Python 2, L{io.StringIO} on Python 3.
    """
    x = _cStringIO(val)
    x.seek(sek)
    return x