from zope.interface import Interface
class IAlias(Interface):
    """
    An interface for aliases.
    """

    def createMessageReceiver():
        """
        Create a message receiver.

        @rtype: L{IMessageSMTP} provider
        @return: A message receiver.
        """