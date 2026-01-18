from twisted.application.internet import StreamServerEndpointService
from twisted.application.service import MultiService
from twisted.conch import telnet
from twisted.cred import error
from twisted.cred.credentials import UsernamePassword
from twisted.python import usage
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_telnetPort(self) -> None:
    """
        L{manhole_tap.makeService} will make a telnet service on the port
        defined by C{--telnetPort}. It will not make a SSH service.
        """
    self.options.parseOptions(['--telnetPort', 'tcp:222'])
    service = manhole_tap.makeService(self.options)
    self.assertIsInstance(service, MultiService)
    self.assertEqual(len(service.services), 1)
    self.assertIsInstance(service.services[0], StreamServerEndpointService)
    self.assertIsInstance(service.services[0].factory.protocol, manhole_tap.makeTelnetProtocol)
    self.assertEqual(service.services[0].endpoint._port, 222)