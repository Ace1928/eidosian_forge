import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def _getEndpoints(self, reactor, service):
    """
        Return a list of endpoints for the specified service, constructing
        defaults if necessary.

        If no endpoints were configured for the service and the protocol
        was not explicitly disabled with a I{--no-*} option, a default
        endpoint for the service is created.

        @type reactor: L{IReactorTCP <twisted.internet.interfaces.IReactorTCP>}
            provider
        @param reactor: If any endpoints are created, the reactor with
            which they are created.

        @type service: L{bytes}
        @param service: The type of service for which to retrieve endpoints,
            either C{b'pop3'} or C{b'smtp'}.

        @rtype: L{list} of L{IStreamServerEndpoint
            <twisted.internet.interfaces.IStreamServerEndpoint>} provider
        @return: The endpoints for the specified service as configured by the
            command line parameters.
        """
    if self[service]:
        return self[service]
    elif self['no-' + service]:
        return []
    else:
        return [endpoints.TCP4ServerEndpoint(reactor, self._protoDefaults[service])]