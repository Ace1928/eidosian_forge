import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class ProtocolIncludingArgument(amp.Argument):
    """
    An L{amp.Argument} which encodes its parser and serializer
    arguments *including the protocol* into its parsed and serialized
    forms.
    """

    def fromStringProto(self, string, protocol):
        """
        Don't decode anything; just return all possible information.

        @return: A two-tuple of the input string and the protocol.
        """
        return (string, protocol)

    def toStringProto(self, obj, protocol):
        """
        Encode identifying information about L{object} and protocol
        into a string for later verification.

        @type obj: L{object}
        @type protocol: L{amp.AMP}
        """
        ident = '%d:%d' % (id(obj), id(protocol))
        return ident.encode('ascii')