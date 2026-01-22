import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class SeqFileRandomAccess(_IndexedSeqFileProxy):
    """Base class for defining random access to sequence files."""

    def __init__(self, filename, format):
        """Initialize the class."""
        self._handle = _open_for_random_access(filename)
        self._format = format
        self._iterator = SeqIO._FormatToIterator[format]

    def get(self, offset):
        """Return SeqRecord."""
        return next(self._iterator(StringIO(self.get_raw(offset).decode())))