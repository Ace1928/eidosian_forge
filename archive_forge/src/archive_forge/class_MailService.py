import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
class MailService(service.MultiService):
    """
    An email service.

    @type queue: L{Queue} or L{None}
    @ivar queue: A queue for outgoing messages.

    @type domains: L{dict} of L{bytes} -> L{IDomain} provider
    @ivar domains: A mapping of supported domain name to domain object.

    @type portals: L{dict} of L{bytes} -> L{Portal}
    @ivar portals: A mapping of domain name to authentication portal.

    @type aliases: L{None} or L{dict} of
        L{bytes} -> L{IAlias} provider
    @ivar aliases: A mapping of domain name to alias.

    @type smtpPortal: L{Portal}
    @ivar smtpPortal: A portal for authentication for the SMTP server.

    @type monitor: L{FileMonitoringService}
    @ivar monitor: A service to monitor changes to files.
    """
    queue = None
    domains = None
    portals = None
    aliases = None
    smtpPortal = None

    def __init__(self):
        """
        Initialize the mail service.
        """
        service.MultiService.__init__(self)
        self.domains = DomainWithDefaultDict({}, BounceDomain())
        self.portals = {}
        self.monitor = FileMonitoringService()
        self.monitor.setServiceParent(self)
        self.smtpPortal = Portal(self)

    def getPOP3Factory(self):
        """
        Create a POP3 protocol factory.

        @rtype: L{POP3Factory}
        @return: A POP3 protocol factory.
        """
        return protocols.POP3Factory(self)

    def getSMTPFactory(self):
        """
        Create an SMTP protocol factory.

        @rtype: L{SMTPFactory <protocols.SMTPFactory>}
        @return: An SMTP protocol factory.
        """
        return protocols.SMTPFactory(self, self.smtpPortal)

    def getESMTPFactory(self):
        """
        Create an ESMTP protocol factory.

        @rtype: L{ESMTPFactory <protocols.ESMTPFactory>}
        @return: An ESMTP protocol factory.
        """
        return protocols.ESMTPFactory(self, self.smtpPortal)

    def addDomain(self, name, domain):
        """
        Add a domain for which the service will accept email.

        @type name: L{bytes}
        @param name: A domain name.

        @type domain: L{IDomain} provider
        @param domain: A domain object.
        """
        portal = Portal(domain)
        map(portal.registerChecker, domain.getCredentialsCheckers())
        self.domains[name] = domain
        self.portals[name] = portal
        if self.aliases and IAliasableDomain.providedBy(domain):
            domain.setAliasGroup(self.aliases)

    def setQueue(self, queue):
        """
        Set the queue for outgoing emails.

        @type queue: L{Queue}
        @param queue: A queue for outgoing messages.
        """
        self.queue = queue

    def requestAvatar(self, avatarId, mind, *interfaces):
        """
        Return a message delivery for an authenticated SMTP user.

        @type avatarId: L{bytes}
        @param avatarId: A string which identifies an authenticated user.

        @type mind: L{None}
        @param mind: Unused.

        @type interfaces: n-L{tuple} of C{zope.interface.Interface}
        @param interfaces: A group of interfaces one of which the avatar must
            support.

        @rtype: 3-L{tuple} of (E{1}) L{IMessageDelivery},
            (E{2}) L{ESMTPDomainDelivery}, (E{3}) no-argument callable
        @return: A tuple of the supported interface, a message delivery, and
            a logout function.

        @raise NotImplementedError: When the given interfaces do not include
            L{IMessageDelivery}.
        """
        if smtp.IMessageDelivery in interfaces:
            a = protocols.ESMTPDomainDelivery(self, avatarId)
            return (smtp.IMessageDelivery, a, lambda: None)
        raise NotImplementedError()

    def lookupPortal(self, name):
        """
        Find the portal for a domain.

        @type name: L{bytes}
        @param name: A domain name.

        @rtype: L{Portal}
        @return: A portal.
        """
        return self.portals[name]

    def defaultPortal(self):
        """
        Return the portal for the default domain.

        The default domain is named ''.

        @rtype: L{Portal}
        @return: The portal for the default domain.
        """
        return self.portals['']