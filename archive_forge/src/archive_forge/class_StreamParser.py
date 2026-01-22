import abc
class StreamParser(object, metaclass=abc.ABCMeta):
    """Streaming parser base class.

    An instance of a subclass of this class is used to extract messages
    from a raw byte stream.

    It's designed to be used for data read from a transport which doesn't
    preserve message boundaries.  A typical example of such a transport
    is TCP.

    """

    class TooSmallException(Exception):
        pass

    def __init__(self):
        self._q = bytearray()

    def parse(self, data):
        """Tries to extract messages from a raw byte stream.

        The data argument would be python bytes newly read from the input
        stream.

        Returns an ordered list of extracted messages.
        It can be an empty list.

        The rest of data which doesn't produce a complete message is
        kept internally and will be used when more data is come.
        I.e. next time this method is called again.
        """
        self._q.append(data)
        msgs = []
        while True:
            try:
                msg, self._q = self.try_parse(self._q)
            except self.TooSmallException:
                break
            msgs.append(msg)
        return msgs

    @abc.abstractmethod
    def try_parse(self, q):
        """Try to extract a message from the given bytes.

        This is an override point for subclasses.

        This method tries to extract a message from bytes given by the
        argument.

        Raises TooSmallException if the given data is not enough to
        extract a complete message but there's still a chance to extract
        a message if more data is come later.
        """
        pass