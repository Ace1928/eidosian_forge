from zope.interface import Interface
class IMessageSMTP(Interface):
    """
    Interface definition for messages that can be sent via SMTP.
    """

    def lineReceived(line):
        """
        Handle another line.
        """

    def eomReceived():
        """
        Handle end of message.

        return a deferred. The deferred should be called with either:
        callback(string) or errback(error)

        @rtype: L{Deferred}
        """

    def connectionLost():
        """
        Handle message truncated.

        semantics should be to discard the message
        """