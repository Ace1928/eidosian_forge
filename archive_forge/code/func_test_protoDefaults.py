from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_protoDefaults(self):
    """
        POP3 and SMTP each listen on a TCP4ServerEndpoint by default.
        """
    options = Options()
    options.parseOptions([])
    self.assertEqual(len(options['pop3']), 1)
    self.assertIsInstance(options['pop3'][0], endpoints.TCP4ServerEndpoint)
    self.assertEqual(len(options['smtp']), 1)
    self.assertIsInstance(options['smtp'][0], endpoints.TCP4ServerEndpoint)