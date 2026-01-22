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
def _nicebytes(bytes):
    """
    Represent a mostly textful bytes object in a way suitable for
    presentation to an end user.

    @param bytes: The bytes to represent.
    @rtype: L{str}
    """
    return repr(bytes)[1:]