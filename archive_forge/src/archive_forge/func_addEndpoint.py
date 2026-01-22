import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def addEndpoint(self, service, description):
    """
        Add an endpoint to a service.

        @type service: L{bytes}
        @param service: A service, either C{b'smtp'} or C{b'pop3'}.

        @type description: L{bytes}
        @param description: An endpoint description string or a TCP port
            number.
        """
    from twisted.internet import reactor
    self[service].append(endpoints.serverFromString(reactor, description))