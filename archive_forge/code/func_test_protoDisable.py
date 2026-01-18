from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def test_protoDisable(self):
    """
        The I{--no-pop3} and I{--no-smtp} options disable POP3 and SMTP
        respectively.
        """
    options = Options()
    options.parseOptions(['--no-pop3'])
    self.assertEqual(options._getEndpoints(None, 'pop3'), [])
    self.assertNotEqual(options._getEndpoints(None, 'smtp'), [])
    options = Options()
    options.parseOptions(['--no-smtp'])
    self.assertNotEqual(options._getEndpoints(None, 'pop3'), [])
    self.assertEqual(options._getEndpoints(None, 'smtp'), [])