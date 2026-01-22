from zope.interface import Interface
class IMessageIMAPFile(Interface):
    """
    Optional message interface for representing messages as files.

    If provided by message objects, this interface will be used instead the
    more complex MIME-based interface.
    """

    def open():
        """
        Return a file-like object opened for reading.

        Reading from the returned file will return all the bytes of which this
        message consists.
        """