import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
@implementer(IAlias)
class FileAlias(AliasBase):
    """
    An alias which translates an address to a file.

    @ivar filename: See L{__init__}.
    """

    def __init__(self, filename, *args):
        """
        @type filename: L{bytes}
        @param filename: The name of the file in which to store the message.

        @type args: 2-L{tuple} of (0) L{dict} mapping L{bytes} to L{IDomain}
            provider, (1) L{bytes}
        @param args: Arguments for L{AliasBase.__init__}.
        """
        AliasBase.__init__(self, *args)
        self.filename = filename

    def __str__(self) -> str:
        """
        Build a string representation of this L{FileAlias} instance.

        @rtype: L{bytes}
        @return: A string containing the name of the file.
        """
        return f'<File {self.filename}>'

    def createMessageReceiver(self):
        """
        Create a message receiver which delivers a message to the file.

        @rtype: L{FileWrapper}
        @return: A message receiver which writes a message to the file.
        """
        return FileWrapper(self.filename)