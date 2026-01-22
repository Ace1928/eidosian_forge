import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
class AliasUpdater:
    """
    A callable object which updates the aliases for a domain from an aliases(5)
    file.

    @ivar domains: See L{__init__}.
    @ivar domain: See L{__init__}.
    """

    def __init__(self, domains, domain):
        """
        @type domains: L{dict} mapping L{bytes} to L{IDomain} provider
        @param domains: A mapping of domain name to domain object

        @type domain: L{IAliasableDomain} provider
        @param domain: The domain to update.
        """
        self.domains = domains
        self.domain = domain

    def __call__(self, new):
        """
        Update the aliases for a domain from an aliases(5) file.

        @type new: L{bytes}
        @param new: The name of an aliases(5) file.
        """
        self.domain.setAliasGroup(alias.loadAliasFile(self.domains, new))