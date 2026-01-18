from twisted.application.internet import StreamServerEndpointService
from twisted.application.service import MultiService
from twisted.conch import telnet
from twisted.cred import error
from twisted.cred.credentials import UsernamePassword
from twisted.python import usage
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_requiresPort(self) -> None:
    """
        L{manhole_tap.makeService} requires either 'telnetPort' or 'sshPort' to
        be given.
        """
    with self.assertRaises(usage.UsageError) as e:
        manhole_tap.Options().parseOptions([])
    self.assertEqual(e.exception.args[0], 'At least one of --telnetPort and --sshPort must be specified')