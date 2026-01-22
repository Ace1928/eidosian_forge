from zope.interface import Interface
class IWriteHandle(Interface):

    def writeToHandle(buff, evt):
        """
        Write the given buffer to this handle.

        @param buff: the buffer to write
        @type buff: any object implementing the buffer protocol

        @param evt: an IOCP Event object

        @return: tuple (return code, number of bytes written)
        """