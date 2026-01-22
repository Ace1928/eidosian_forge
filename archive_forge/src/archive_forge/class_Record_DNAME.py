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
class Record_DNAME(SimpleRecord):
    """
    A non-terminal DNS name redirection.

    This record type provides the capability to map an entire subtree of the
    DNS name space to another domain.  It differs from the CNAME record which
    maps a single node of the name space.

    @see: U{http://www.faqs.org/rfcs/rfc2672.html}
    @see: U{http://www.faqs.org/rfcs/rfc3363.html}
    """
    TYPE = DNAME
    fancybasename = 'DNAME'