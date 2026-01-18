from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase
def test_secondaryExplicitPort(self) -> None:
    """
        An argument of the form C{"ip:port/domain"} can be used to specify an
        alternate port number for which to act as a secondary.
        """
    options = Options()
    options.parseOptions(['--secondary', '1.2.3.4:5353/example.com'])
    self.assertEqual([(('1.2.3.4', 5353), ['example.com'])], options.secondaries)