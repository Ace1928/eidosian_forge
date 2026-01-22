import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class SffTrimedRandomAccess(SffRandomAccess):
    """Random access to an SFF file with defined trimming applied to each sequence."""

    def get(self, offset):
        """Return the SeqRecord starting at the given offset."""
        handle = self._handle
        handle.seek(offset)
        return SeqIO.SffIO._sff_read_seq_record(handle, self._flows_per_read, self._flow_chars, self._key_sequence, trim=True)