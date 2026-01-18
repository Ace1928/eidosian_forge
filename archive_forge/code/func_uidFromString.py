from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def uidFromString(uidString):
    """
    Convert a user identifier, as a string, into an integer UID.

    @type uidString: C{str}
    @param uidString: A string giving the base-ten representation of a UID or
        the name of a user which can be converted to a UID via L{pwd.getpwnam}.

    @rtype: C{int}
    @return: The integer UID corresponding to the given string.

    @raise ValueError: If the user name is supplied and L{pwd} is not
        available.
    """
    try:
        return int(uidString)
    except ValueError:
        if pwd is None:
            raise
        return pwd.getpwnam(uidString)[2]