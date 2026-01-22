import struct
import sys
import zlib
from builtins import open as _open
class BgzfWriter:
    """Define a BGZFWriter object."""

    def __init__(self, filename=None, mode='w', fileobj=None, compresslevel=6):
        """Initilize the class."""
        if filename and fileobj:
            raise ValueError('Supply either filename or fileobj, not both')
        if fileobj:
            if fileobj.read(0) != b'':
                raise ValueError('fileobj not opened in binary mode')
            handle = fileobj
        else:
            if 'w' not in mode.lower() and 'a' not in mode.lower():
                raise ValueError(f'Must use write or append mode, not {mode!r}')
            if 'a' in mode.lower():
                handle = _open(filename, 'ab')
            else:
                handle = _open(filename, 'wb')
        self._text = 'b' not in mode.lower()
        self._handle = handle
        self._buffer = b''
        self.compresslevel = compresslevel

    def _write_block(self, block):
        """Write provided data to file as a single BGZF compressed block (PRIVATE)."""
        if len(block) > 65536:
            raise ValueError(f'{len(block)} Block length > 65536')
        c = zlib.compressobj(self.compresslevel, zlib.DEFLATED, -15, zlib.DEF_MEM_LEVEL, 0)
        compressed = c.compress(block) + c.flush()
        del c
        if len(compressed) > 65536:
            raise RuntimeError("TODO - Didn't compress enough, try less data in this block")
        crc = zlib.crc32(block)
        if crc < 0:
            crc = struct.pack('<i', crc)
        else:
            crc = struct.pack('<I', crc)
        bsize = struct.pack('<H', len(compressed) + 25)
        crc = struct.pack('<I', zlib.crc32(block) & 4294967295)
        uncompressed_length = struct.pack('<I', len(block))
        data = _bgzf_header + bsize + compressed + crc + uncompressed_length
        self._handle.write(data)

    def write(self, data):
        """Write method for the class."""
        if isinstance(data, str):
            data = data.encode('latin-1')
        data_len = len(data)
        if len(self._buffer) + data_len < 65536:
            self._buffer += data
        else:
            self._buffer += data
            while len(self._buffer) >= 65536:
                self._write_block(self._buffer[:65536])
                self._buffer = self._buffer[65536:]

    def flush(self):
        """Flush data explicitally."""
        while len(self._buffer) >= 65536:
            self._write_block(self._buffer[:65535])
            self._buffer = self._buffer[65535:]
        self._write_block(self._buffer)
        self._buffer = b''
        self._handle.flush()

    def close(self):
        """Flush data, write 28 bytes BGZF EOF marker, and close BGZF file.

        samtools will look for a magic EOF marker, just a 28 byte empty BGZF
        block, and if it is missing warns the BAM file may be truncated. In
        addition to samtools writing this block, so too does bgzip - so this
        implementation does too.
        """
        if self._buffer:
            self.flush()
        self._handle.write(_bgzf_eof)
        self._handle.flush()
        self._handle.close()

    def tell(self):
        """Return a BGZF 64-bit virtual offset."""
        return make_virtual_offset(self._handle.tell(), len(self._buffer))

    def seekable(self):
        """Return True indicating the BGZF supports random access."""
        return False

    def isatty(self):
        """Return True if connected to a TTY device."""
        return False

    def fileno(self):
        """Return integer file descriptor."""
        return self._handle.fileno()

    def __enter__(self):
        """Open a file operable with WITH statement."""
        return self

    def __exit__(self, type, value, traceback):
        """Close a file with WITH statement."""
        self.close()