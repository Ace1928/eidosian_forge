import sys
import os
import struct
import logging
import numpy as np
def _read_undefined_length_value(self, read_size=128):
    """Copied (in compacted form) from PyDicom
        Copyright Darcy Mason.
        """
    fp = self._file
    search_rewind = 3
    bytes_to_find = struct.pack(self._unpackPrefix + 'HH', SequenceDelimiterTag[0], SequenceDelimiterTag[1])
    found = False
    value_chunks = []
    while not found:
        chunk_start = fp.tell()
        bytes_read = fp.read(read_size)
        if len(bytes_read) < read_size:
            new_bytes = fp.read(read_size - len(bytes_read))
            bytes_read += new_bytes
            if len(bytes_read) < read_size:
                raise EOFError('End of file reached before sequence delimiter found.')
        index = bytes_read.find(bytes_to_find)
        if index != -1:
            found = True
            value_chunks.append(bytes_read[:index])
            fp.seek(chunk_start + index + 4)
            length = fp.read(4)
            if length != b'\x00\x00\x00\x00':
                logger.warning('Expected 4 zero bytes after undefined length delimiter')
        else:
            fp.seek(fp.tell() - search_rewind)
            value_chunks.append(bytes_read[:-search_rewind])
    return b''.join(value_chunks)