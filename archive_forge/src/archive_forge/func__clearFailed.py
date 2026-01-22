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
def _clearFailed(self, deferred, id):
    """
        Clean the Deferred after a timeout.
        """
    try:
        del self.liveMessages[id]
    except KeyError:
        pass
    deferred.errback(failure.Failure(DNSQueryTimeoutError(id)))