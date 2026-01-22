from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
class BufferedPktLineWriter:
    """Writer that wraps its data in pkt-lines and has an independent buffer.

    Consecutive calls to write() wrap the data in a pkt-line and then buffers
    it until enough lines have been written such that their total length
    (including length prefix) reach the buffer size.
    """

    def __init__(self, write, bufsize=65515) -> None:
        """Initialize the BufferedPktLineWriter.

        Args:
          write: A write callback for the underlying writer.
          bufsize: The internal buffer size, including length prefixes.
        """
        self._write = write
        self._bufsize = bufsize
        self._wbuf = BytesIO()
        self._buflen = 0

    def write(self, data):
        """Write data, wrapping it in a pkt-line."""
        line = pkt_line(data)
        line_len = len(line)
        over = self._buflen + line_len - self._bufsize
        if over >= 0:
            start = line_len - over
            self._wbuf.write(line[:start])
            self.flush()
        else:
            start = 0
        saved = line[start:]
        self._wbuf.write(saved)
        self._buflen += len(saved)

    def flush(self):
        """Flush all data from the buffer."""
        data = self._wbuf.getvalue()
        if data:
            self._write(data)
        self._len = 0
        self._wbuf = BytesIO()