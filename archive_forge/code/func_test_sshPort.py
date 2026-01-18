from twisted.application.internet import StreamServerEndpointService
from twisted.application.service import MultiService
from twisted.conch import telnet
from twisted.cred import error
from twisted.cred.credentials import UsernamePassword
from twisted.python import usage
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_sshPort(self) -> None:
    """
        L{manhole_tap.makeService} will make a SSH service on the port
        defined by C{--sshPort}. It will not make a telnet service.
        """
    self.options.parseOptions(['--sshKeyDir', self.mktemp(), '--sshKeySize', '1024', '--sshPort', 'tcp:223'])
    service = manhole_tap.makeService(self.options)
    self.assertIsInstance(service, MultiService)
    self.assertEqual(len(service.services), 1)
    self.assertIsInstance(service.services[0], StreamServerEndpointService)
    self.assertIsInstance(service.services[0].factory, manhole_ssh.ConchFactory)
    self.assertEqual(service.services[0].endpoint._port, 223)