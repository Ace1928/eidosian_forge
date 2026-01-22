import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
@implementer(mail.IAliasableDomain)
class AbstractMaildirDomain:
    """
    An abstract maildir-backed domain.

    @type alias: L{None} or L{dict} mapping
        L{bytes} to L{AliasBase}
    @ivar alias: A mapping of username to alias.

    @ivar root: See L{__init__}.
    """
    alias = None
    root = None

    def __init__(self, service, root):
        """
        @type service: L{MailService}
        @param service: An email service.

        @type root: L{bytes}
        @param root: The maildir root directory.
        """
        self.root = root

    def userDirectory(self, user):
        """
        Return the maildir directory for a user.

        @type user: L{bytes}
        @param user: A username.

        @rtype: L{bytes} or L{None}
        @return: The user's mail directory for a valid user. Otherwise,
            L{None}.
        """
        return None

    def setAliasGroup(self, alias):
        """
        Set the group of defined aliases for this domain.

        @type alias: L{dict} mapping L{bytes} to L{IAlias} provider.
        @param alias: A mapping of domain name to alias.
        """
        self.alias = alias

    def exists(self, user, memo=None):
        """
        Check whether a user exists in this domain or an alias of it.

        @type user: L{User}
        @param user: A user.

        @type memo: L{None} or L{dict} of L{AliasBase}
        @param memo: A record of the addresses already considered while
            resolving aliases. The default value should be used by all
            external code.

        @rtype: no-argument callable which returns L{IMessage <smtp.IMessage>}
            provider.
        @return: A function which takes no arguments and returns a message
            receiver for the user.

        @raises SMTPBadRcpt: When the given user does not exist in this domain
            or an alias of it.
        """
        if self.userDirectory(user.dest.local) is not None:
            return lambda: self.startMessage(user)
        try:
            a = self.alias[user.dest.local]
        except BaseException:
            raise smtp.SMTPBadRcpt(user)
        else:
            aliases = a.resolve(self.alias, memo)
            if aliases:
                return lambda: aliases
            log.err('Bad alias configuration: ' + str(user))
            raise smtp.SMTPBadRcpt(user)

    def startMessage(self, user):
        """
        Create a maildir message for a user.

        @type user: L{bytes}
        @param user: A username.

        @rtype: L{MaildirMessage}
        @return: A message receiver for this user.
        """
        if isinstance(user, str):
            name, domain = user.split('@', 1)
        else:
            name, domain = (user.dest.local, user.dest.domain)
        dir = self.userDirectory(name)
        fname = _generateMaildirName()
        filename = os.path.join(dir, 'tmp', fname)
        fp = open(filename, 'w')
        return MaildirMessage(f'{name}@{domain}', fp, filename, os.path.join(dir, 'new', fname))

    def willRelay(self, user, protocol):
        """
        Check whether this domain will relay.

        @type user: L{Address}
        @param user: The destination address.

        @type protocol: L{SMTP}
        @param protocol: The protocol over which the message to be relayed is
            being received.

        @rtype: L{bool}
        @return: An indication of whether this domain will relay the message to
            the destination.
        """
        return False

    def addUser(self, user, password):
        """
        Add a user to this domain.

        Subclasses should override this method.

        @type user: L{bytes}
        @param user: A username.

        @type password: L{bytes}
        @param password: A password.
        """
        raise NotImplementedError

    def getCredentialsCheckers(self):
        """
        Return credentials checkers for this domain.

        Subclasses should override this method.

        @rtype: L{list} of L{ICredentialsChecker
            <checkers.ICredentialsChecker>} provider
        @return: Credentials checkers for this domain.
        """
        raise NotImplementedError