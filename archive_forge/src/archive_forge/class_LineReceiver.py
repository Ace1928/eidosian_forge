import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class LineReceiver(protocol.Protocol, _PauseableMixin):
    """
    A protocol that receives lines and/or raw data, depending on mode.

    In line mode, each line that's received becomes a callback to
    L{lineReceived}.  In raw data mode, each chunk of raw data becomes a
    callback to L{LineReceiver.rawDataReceived}.
    The L{setLineMode} and L{setRawMode} methods switch between the two modes.

    This is useful for line-oriented protocols such as IRC, HTTP, POP, etc.

    @cvar delimiter: The line-ending delimiter to use. By default this is
                     C{b'\\r\\n'}.
    @cvar MAX_LENGTH: The maximum length of a line to allow (If a
                      sent line is longer than this, the connection is dropped).
                      Default is 16384.
    """
    line_mode = 1
    _buffer = b''
    _busyReceiving = False
    delimiter = b'\r\n'
    MAX_LENGTH = 16384

    def clearLineBuffer(self):
        """
        Clear buffered data.

        @return: All of the cleared buffered data.
        @rtype: C{bytes}
        """
        b, self._buffer = (self._buffer, b'')
        return b

    def dataReceived(self, data):
        """
        Protocol.dataReceived.
        Translates bytes into lines, and calls lineReceived (or
        rawDataReceived, depending on mode.)
        """
        if self._busyReceiving:
            self._buffer += data
            return
        try:
            self._busyReceiving = True
            self._buffer += data
            while self._buffer and (not self.paused):
                if self.line_mode:
                    try:
                        line, self._buffer = self._buffer.split(self.delimiter, 1)
                    except ValueError:
                        if len(self._buffer) >= self.MAX_LENGTH + len(self.delimiter):
                            line, self._buffer = (self._buffer, b'')
                            return self.lineLengthExceeded(line)
                        return
                    else:
                        lineLength = len(line)
                        if lineLength > self.MAX_LENGTH:
                            exceeded = line + self.delimiter + self._buffer
                            self._buffer = b''
                            return self.lineLengthExceeded(exceeded)
                        why = self.lineReceived(line)
                        if why or (self.transport and self.transport.disconnecting):
                            return why
                else:
                    data = self._buffer
                    self._buffer = b''
                    why = self.rawDataReceived(data)
                    if why:
                        return why
        finally:
            self._busyReceiving = False

    def setLineMode(self, extra=b''):
        """
        Sets the line-mode of this receiver.

        If you are calling this from a rawDataReceived callback,
        you can pass in extra unhandled data, and that data will
        be parsed for lines.  Further data received will be sent
        to lineReceived rather than rawDataReceived.

        Do not pass extra data if calling this function from
        within a lineReceived callback.
        """
        self.line_mode = 1
        if extra:
            return self.dataReceived(extra)

    def setRawMode(self):
        """
        Sets the raw mode of this receiver.
        Further data received will be sent to rawDataReceived rather
        than lineReceived.
        """
        self.line_mode = 0

    def rawDataReceived(self, data):
        """
        Override this for when raw data is received.
        """
        raise NotImplementedError

    def lineReceived(self, line):
        """
        Override this for when each line is received.

        @param line: The line which was received with the delimiter removed.
        @type line: C{bytes}
        """
        raise NotImplementedError

    def sendLine(self, line):
        """
        Sends a line to the other end of the connection.

        @param line: The line to send, not including the delimiter.
        @type line: C{bytes}
        """
        return self.transport.write(line + self.delimiter)

    def lineLengthExceeded(self, line):
        """
        Called when the maximum line length has been reached.
        Override if it needs to be dealt with in some special way.

        The argument 'line' contains the remainder of the buffer, starting
        with (at least some part) of the line which is too long. This may
        be more than one line, or may be only the initial portion of the
        line.
        """
        return self.transport.loseConnection()