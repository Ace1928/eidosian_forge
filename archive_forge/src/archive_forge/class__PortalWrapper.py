import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
class _PortalWrapper(Referenceable, _JellyableAvatarMixin):
    """
    Root Referenceable object, used to login to portal.
    """

    def __init__(self, portal, broker):
        self.portal = portal
        self.broker = broker

    def remote_login(self, username):
        """
        Start of username/password login.

        @param username: The username.
        """
        c = challenge()
        return (c, _PortalAuthChallenger(self.portal, self.broker, username, c))

    def remote_loginAnonymous(self, mind):
        """
        Attempt an anonymous login.

        @param mind: An object to use as the mind parameter to the portal login
            call (possibly None).

        @rtype: L{Deferred}
        @return: A Deferred which will be called back with an avatar when login
            succeeds or which will be errbacked if login fails somehow.
        """
        d = self.portal.login(Anonymous(), mind, IPerspective)
        d.addCallback(self._cbLogin)
        return d