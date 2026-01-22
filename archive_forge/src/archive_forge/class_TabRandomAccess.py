import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class TabRandomAccess(SeqFileRandomAccess):
    """Random access to a simple tabbed file."""

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        tab_char = b'\t'
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                key = line.split(tab_char)[0]
            except ValueError:
                if not line.strip():
                    continue
                else:
                    raise
            else:
                yield (key.decode(), start_offset, len(line))

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        handle.seek(offset)
        return handle.readline()