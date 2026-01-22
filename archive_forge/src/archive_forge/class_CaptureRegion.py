import struct
from oslo_log import log as logging
class CaptureRegion(object):
    """Represents a region of a file we want to capture.

    A region of a file we want to capture requires a byte offset into
    the file and a length. This is expected to be used by a data
    processing loop, calling capture() with the most recently-read
    chunk. This class handles the task of grabbing the desired region
    of data across potentially multiple fractional and unaligned reads.

    :param offset: Byte offset into the file starting the region
    :param length: The length of the region
    """

    def __init__(self, offset, length):
        self.offset = offset
        self.length = length
        self.data = b''

    @property
    def complete(self):
        """Returns True when we have captured the desired data."""
        return self.length == len(self.data)

    def capture(self, chunk, current_position):
        """Process a chunk of data.

        This should be called for each chunk in the read loop, at least
        until complete returns True.

        :param chunk: A chunk of bytes in the file
        :param current_position: The position of the file processed by the
                                 read loop so far. Note that this will be
                                 the position in the file *after* the chunk
                                 being presented.
        """
        read_start = current_position - len(chunk)
        if read_start <= self.offset <= current_position or self.offset <= read_start <= self.offset + self.length:
            if read_start < self.offset:
                lead_gap = self.offset - read_start
            else:
                lead_gap = 0
            self.data += chunk[lead_gap:]
            self.data = self.data[:self.length]