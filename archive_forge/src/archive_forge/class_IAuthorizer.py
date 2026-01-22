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
class IAuthorizer(Interface):

    def getChallenge(peer):
        """
        Generate a challenge the client may respond to.

        @type peer: C{tuple}
        @param peer: The client's address

        @rtype: C{str}
        @return: The challenge string
        """

    def decode(response):
        """
        Create a credentials object from the given response.

        @type response: C{str}
        """