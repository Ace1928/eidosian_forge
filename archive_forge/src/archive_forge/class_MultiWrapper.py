import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
@implementer(smtp.IMessage)
class MultiWrapper:
    """
    A message receiver which delivers a single message to multiple other
    message receivers.

    @ivar objs: See L{__init__}.
    """

    def __init__(self, objs):
        """
        @type objs: L{list} of L{IMessage <smtp.IMessage>} provider
        @param objs: Message receivers to which the incoming message should be
            directed.
        """
        self.objs = objs

    def lineReceived(self, line):
        """
        Pass a received line to the message receivers.

        @type line: L{bytes}
        @param line: A line of the message.
        """
        for o in self.objs:
            o.lineReceived(line)

    def eomReceived(self):
        """
        Pass the end of message along to the message receivers.

        @rtype: L{DeferredList <defer.DeferredList>} whose successful results
            are L{bytes} or L{None}
        @return: A deferred list which triggers when all of the message
            receivers have finished handling their end of message.
        """
        return defer.DeferredList([o.eomReceived() for o in self.objs])

    def connectionLost(self):
        """
        Inform the message receivers that the connection has been lost.
        """
        for o in self.objs:
            o.connectionLost()

    def __str__(self) -> str:
        """
        Build a string representation of this L{MultiWrapper} instance.

        @rtype: L{bytes}
        @return: A string containing a list of the message receivers.
        """
        return f'<GroupWrapper {map(str, self.objs)!r}>'