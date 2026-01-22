import array
import struct
from . import errors
from .io import gfile
class PyRecordReader_New:

    def __init__(self, filename=None, start_offset=0, compression_type=None, status=None):
        if filename is None:
            raise errors.NotFoundError(None, None, 'No filename provided, cannot read Events')
        if not gfile.exists(filename):
            raise errors.NotFoundError(None, None, '{} does not point to valid Events file'.format(filename))
        if start_offset:
            raise errors.UnimplementedError(None, None, 'start offset not supported by compat reader')
        if compression_type:
            raise errors.UnimplementedError(None, None, 'compression not supported by compat reader')
        self.filename = filename
        self.start_offset = start_offset
        self.compression_type = compression_type
        self.status = status
        self.curr_event = None
        self.file_handle = gfile.GFile(self.filename, 'rb')
        self._buffer = b''
        self._buffer_pos = 0

    def GetNext(self):
        self._buffer_pos = 0
        self.curr_event = None
        header_str = self._read(8)
        if not header_str:
            raise errors.OutOfRangeError(None, None, 'No more events to read')
        if len(header_str) < 8:
            raise self._truncation_error('header')
        header = struct.unpack('<Q', header_str)
        crc_header_str = self._read(4)
        if len(crc_header_str) < 4:
            raise self._truncation_error('header crc')
        crc_header = struct.unpack('<I', crc_header_str)
        header_crc_calc = masked_crc32c(header_str)
        if header_crc_calc != crc_header[0]:
            raise errors.DataLossError(None, None, '{} failed header crc32 check'.format(self.filename))
        header_len = int(header[0])
        event_str = self._read(header_len)
        if len(event_str) < header_len:
            raise self._truncation_error('data')
        event_crc_calc = masked_crc32c(event_str)
        crc_event_str = self._read(4)
        if len(crc_event_str) < 4:
            raise self._truncation_error('data crc')
        crc_event = struct.unpack('<I', crc_event_str)
        if event_crc_calc != crc_event[0]:
            raise errors.DataLossError(None, None, '{} failed event crc32 check'.format(self.filename))
        self.curr_event = event_str
        self._buffer = b''

    def _read(self, n):
        """Read up to n bytes from the underlying file, with buffering.

        Reads are satisfied from a buffer of previous data read starting at
        `self._buffer_pos` until the buffer is exhausted, and then from the
        actual underlying file. Any new data is added to the buffer, and
        `self._buffer_pos` is advanced to the point in the buffer past all
        data returned as part of this read.

        Args:
          n: non-negative number of bytes to read

        Returns:
          bytestring of data read, up to n bytes
        """
        result = self._buffer[self._buffer_pos:self._buffer_pos + n]
        self._buffer_pos += len(result)
        n -= len(result)
        if n > 0:
            new_data = self.file_handle.read(n)
            result += new_data
            self._buffer += new_data
            self._buffer_pos += len(new_data)
        return result

    def _truncation_error(self, section):
        return errors.DataLossError(None, None, '{} has truncated record in {}'.format(self.filename, section))

    def record(self):
        return self.curr_event