from zope.interface import Interface
class IClientAuthentication(Interface):

    def getName():
        """
        Return an identifier associated with this authentication scheme.

        @rtype: L{bytes}
        """

    def challengeResponse(secret, challenge):
        """
        Generate a challenge response string.
        """