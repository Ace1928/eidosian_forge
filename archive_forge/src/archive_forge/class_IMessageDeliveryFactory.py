from zope.interface import Interface
class IMessageDeliveryFactory(Interface):
    """
    An alternate interface to implement for handling message delivery.

    It is useful to implement this interface instead of L{IMessageDelivery}
    directly because it allows the implementor to distinguish between different
    messages delivery over the same connection. This can be used to optimize
    delivery of a single message to multiple recipients, something which cannot
    be done by L{IMessageDelivery} implementors due to their lack of
    information.
    """

    def getMessageDelivery():
        """
        Return an L{IMessageDelivery} object.

        This will be called once per message.
        """