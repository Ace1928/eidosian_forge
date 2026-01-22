import re
from io import BytesIO
from .. import errors
class BytesRecordReader(BaseReader):

    def read(self):
        """Read this record.

        You can either validate or read a record, you can't do both.

        :returns: A tuple of (names, callable).  The callable can be called
            repeatedly to obtain the bytes for the record, with a max_length
            argument.  If max_length is None, returns all the bytes.  Because
            records can be arbitrarily large, using None is not recommended
            unless you have reason to believe the content will fit in memory.
        """
        length_line = self._read_line()
        try:
            length = int(length_line)
        except ValueError:
            raise InvalidRecordError('{!r} is not a valid length.'.format(length_line))
        names = []
        while True:
            name_line = self._read_line()
            if name_line == b'':
                break
            name_tuple = tuple(name_line.split(b'\x00'))
            for name in name_tuple:
                _check_name(name)
            names.append(name_tuple)
        self._remaining_length = length
        return (names, self._content_reader)

    def _content_reader(self, max_length):
        if max_length is None:
            length_to_read = self._remaining_length
        else:
            length_to_read = min(max_length, self._remaining_length)
        self._remaining_length -= length_to_read
        bytes = self.reader_func(length_to_read)
        if len(bytes) != length_to_read:
            raise UnexpectedEndOfContainerError()
        return bytes

    def validate(self):
        """Validate this record.

        You can either validate or read, you can't do both.

        :raises ContainerError: if this record is invalid.
        """
        names, read_bytes = self.read()
        for name_tuple in names:
            for name in name_tuple:
                _check_name_encoding(name)
        read_bytes(None)