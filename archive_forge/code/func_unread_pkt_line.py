from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def unread_pkt_line(self, data):
    """Unread a single line of data into the readahead buffer.

        This method can be used to unread a single pkt-line into a fixed
        readahead buffer.

        Args:
          data: The data to unread, without the length prefix.

        Raises:
          ValueError: If more than one pkt-line is unread.
        """
    if self._readahead is not None:
        raise ValueError('Attempted to unread multiple pkt-lines.')
    self._readahead = BytesIO(pkt_line(data))