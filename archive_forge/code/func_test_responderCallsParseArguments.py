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
def test_responderCallsParseArguments(self):
    """
        Making a remote call on a L{amp.Command} subclass which
        overrides the C{parseArguments} method should call that
        C{parseArguments} method to get the arguments.
        """
    protocol = NoNetworkProtocol()
    responder = protocol.locateResponder(MagicSchemaCommand.commandName)
    argument = object()
    response = responder(dict(weird=argument))
    response.addCallback(lambda ign: self.assertEqual(protocol.parseArgumentsArguments, ({'weird': argument}, protocol)))
    return response