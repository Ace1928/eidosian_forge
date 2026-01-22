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
def _nicebyteslist(list):
    """
    Represent a list of mostly textful bytes objects in a way suitable for
    presentation to an end user.

    @param list: The list of bytes to represent.
    @rtype: L{str}
    """
    return '[{}]'.format(', '.join([_nicebytes(b) for b in list]))