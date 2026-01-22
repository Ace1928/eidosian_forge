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
def _isSubdomainOf(descendantName, ancestorName):
    """
    Test whether C{descendantName} is equal to or is a I{subdomain} of
    C{ancestorName}.

    The names are compared case-insensitively.

    The names are treated as byte strings containing one or more
    DNS labels separated by B{.}.

    C{descendantName} is considered equal if its sequence of labels
    exactly matches the labels of C{ancestorName}.

    C{descendantName} is considered a I{subdomain} if its sequence of
    labels ends with the labels of C{ancestorName}.

    @type descendantName: L{bytes}
    @param descendantName: The DNS subdomain name.

    @type ancestorName: L{bytes}
    @param ancestorName: The DNS parent or ancestor domain name.

    @return: C{True} if C{descendantName} is equal to or if it is a
        subdomain of C{ancestorName}. Otherwise returns C{False}.
    """
    descendantLabels = _nameToLabels(descendantName.lower())
    ancestorLabels = _nameToLabels(ancestorName.lower())
    return descendantLabels[-len(ancestorLabels):] == ancestorLabels