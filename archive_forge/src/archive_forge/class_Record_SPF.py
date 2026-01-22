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
class Record_SPF(Record_TXT):
    """
    Structurally, freeform text. Semantically, a policy definition, formatted
    as defined in U{rfc 4408<http://www.faqs.org/rfcs/rfc4408.html>}.

    @type data: L{list} of L{bytes}
    @ivar data: Freeform text which makes up this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds
               which this record should be cached.
    """
    TYPE = SPF
    fancybasename = 'SPF'