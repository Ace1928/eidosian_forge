from twisted.application import internet, strports
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet.protocol import Factory
from twisted.trial.unittest import TestCase
def test_serviceDefaultReactor(self):
    """
        L{strports.service} will use the default reactor when none is provided
        as an argument.
        """
    from twisted.internet import reactor as globalReactor
    aService = strports.service('tcp:80', None)
    self.assertIs(aService.endpoint._reactor, globalReactor)