import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class IntelliGeneticsRandomAccess(SeqFileRandomAccess):
    """Random access to a IntelliGenetics file."""

    def __init__(self, filename, format):
        """Initialize the class."""
        SeqFileRandomAccess.__init__(self, filename, format)
        self._marker_re = re.compile(b'^;')

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        offset = 0
        line = ''
        while True:
            offset += len(line)
            line = handle.readline()
            if not line:
                break
            if not line.startswith(b';;'):
                break
        while line:
            length = 0
            assert offset + len(line) == handle.tell()
            if not line.startswith(b';'):
                raise ValueError(f"Records should start with ';' and not:\n{line!r}")
            while line.startswith(b';'):
                length += len(line)
                line = handle.readline()
            key = line.rstrip()
            while line and (not line.startswith(b';')):
                length += len(line)
                line = handle.readline()
            yield (key.decode(), offset, length)
            offset += length
            assert offset + len(line) == handle.tell()

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        handle.seek(offset)
        marker_re = self._marker_re
        lines = []
        line = handle.readline()
        while line.startswith(b';'):
            lines.append(line)
            line = handle.readline()
        while line and (not line.startswith(b';')):
            lines.append(line)
            line = handle.readline()
        return b''.join(lines)