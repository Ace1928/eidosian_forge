from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_smtp(self):
    """
        If one or more endpoints is included in the configuration passed to
        L{makeService} for the C{"smtp"} key, a service for starting an SMTP
        server is constructed for each of them and attached to the returned
        service.
        """
    self._endpointServerTest('smtp', protocols.SMTPFactory)