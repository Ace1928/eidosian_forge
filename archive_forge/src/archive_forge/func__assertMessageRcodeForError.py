from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def _assertMessageRcodeForError(self, responseError, expectedMessageCode):
    """
        L{server.DNSServerFactory.gotResolver} accepts a L{failure.Failure} and
        triggers a response message whose rCode corresponds to the DNS error
        contained in the C{Failure}.

        @param responseError: The L{Exception} instance which is expected to
            trigger C{expectedMessageCode} when it is supplied to
            C{gotResolverError}
        @type responseError: L{Exception}

        @param expectedMessageCode: The C{rCode} which is expected in the
            message returned by C{gotResolverError} in response to
            C{responseError}.
        @type expectedMessageCode: L{int}
        """
    f = server.DNSServerFactory()
    e = self.assertRaises(RaisingProtocol.WriteMessageArguments, f.gotResolverError, failure.Failure(responseError), protocol=RaisingProtocol(), message=dns.Message(), address=None)
    (message,), kwargs = e.args
    self.assertEqual(message.rCode, expectedMessageCode)