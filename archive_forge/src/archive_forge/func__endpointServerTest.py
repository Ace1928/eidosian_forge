from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def _endpointServerTest(self, key, factoryClass):
    """
        Configure a service with two endpoints for the protocol associated with
        C{key} and verify that when the service is started a factory of type
        C{factoryClass} is used to listen on each of them.
        """
    cleartext = SpyEndpoint()
    secure = SpyEndpoint()
    config = Options()
    config[key] = [cleartext, secure]
    service = makeService(config)
    service.privilegedStartService()
    service.startService()
    self.addCleanup(service.stopService)
    self.assertIsInstance(cleartext.listeningWith, factoryClass)
    self.assertIsInstance(secure.listeningWith, factoryClass)