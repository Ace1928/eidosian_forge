import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
class ILocator(Interface):
    """
    Allow looking up physical address for logical URL.
    """

    def getAddress(logicalURL):
        """
        Return physical URL of server for logical URL of user.

        @param logicalURL: a logical C{URL}.
        @return: Deferred which becomes URL or fails with LookupError.
        """