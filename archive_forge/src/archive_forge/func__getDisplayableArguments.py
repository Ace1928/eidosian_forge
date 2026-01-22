from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def _getDisplayableArguments(obj, alwaysShow, fieldNames):
    """
    Inspect the function signature of C{obj}'s constructor,
    and get a list of which arguments should be displayed.
    This is a helper function for C{_compactRepr}.

    @param obj: The instance whose repr is being generated.
    @param alwaysShow: A L{list} of field names which should always be shown.
    @param fieldNames: A L{list} of field attribute names which should be shown
        if they have non-default values.
    @return: A L{list} of displayable arguments.
    """
    displayableArgs = []
    signature = inspect.signature(obj.__class__.__init__)
    for name in fieldNames:
        defaultValue = signature.parameters[name].default
        fieldValue = getattr(obj, name, defaultValue)
        if name in alwaysShow or fieldValue != defaultValue:
            displayableArgs.append(f' {name}={fieldValue!r}')
    return displayableArgs