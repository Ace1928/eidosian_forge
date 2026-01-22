from zope.interface import Interface
class IMessageIMAP(IMessageIMAPPart):

    def getUID():
        """
        Retrieve the unique identifier associated with this message.
        """

    def getFlags():
        """
        Retrieve the flags associated with this message.

        @rtype: C{iterable}
        @return: The flags, represented as strings.
        """

    def getInternalDate():
        """
        Retrieve the date internally associated with this message.

        @rtype: L{bytes}
        @return: An RFC822-formatted date string.
        """