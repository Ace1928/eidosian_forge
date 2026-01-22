from zope.interface import Attribute, Interface
class IProtocolPlugin(Interface):
    """Interface for plugins providing an interface to a Words service"""
    name = Attribute('A single word describing what kind of interface this is (eg, irc or web)')

    def getFactory(realm, portal):
        """Retrieve a C{twisted.internet.interfaces.IServerFactory} provider

        @param realm: An object providing C{twisted.cred.portal.IRealm} and
        L{IChatService}, with which service information should be looked up.

        @param portal: An object providing C{twisted.cred.portal.IPortal},
        through which logins should be performed.
        """