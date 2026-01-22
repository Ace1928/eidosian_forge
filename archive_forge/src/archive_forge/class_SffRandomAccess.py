import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class SffRandomAccess(SeqFileRandomAccess):
    """Random access to a Standard Flowgram Format (SFF) file."""

    def __init__(self, filename, format):
        """Initialize the class."""
        SeqFileRandomAccess.__init__(self, filename, format)
        header_length, index_offset, index_length, number_of_reads, self._flows_per_read, self._flow_chars, self._key_sequence = SeqIO.SffIO._sff_file_header(self._handle)

    def __iter__(self):
        """Load any index block in the file, or build it the slow way (PRIVATE)."""
        handle = self._handle
        handle.seek(0)
        header_length, index_offset, index_length, number_of_reads, self._flows_per_read, self._flow_chars, self._key_sequence = SeqIO.SffIO._sff_file_header(handle)
        if index_offset and index_length:
            count = 0
            max_offset = 0
            try:
                for name, offset in SeqIO.SffIO._sff_read_roche_index(handle):
                    max_offset = max(max_offset, offset)
                    yield (name, offset, 0)
                    count += 1
                if count != number_of_reads:
                    raise ValueError('Indexed %i records, expected %i' % (count, number_of_reads))
            except ValueError as err:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn(f'Could not parse the SFF index: {err}', BiopythonParserWarning)
                assert count == 0, 'Partially populated index'
                handle.seek(0)
            else:
                if index_offset + index_length <= max_offset:
                    handle.seek(max_offset)
                    SeqIO.SffIO._sff_read_raw_record(handle, self._flows_per_read)
                SeqIO.SffIO._check_eof(handle, index_offset, index_length)
                return
        count = 0
        for name, offset in SeqIO.SffIO._sff_do_slow_index(handle):
            yield (name, offset, 0)
            count += 1
        if count != number_of_reads:
            raise ValueError('Indexed %i records, expected %i' % (count, number_of_reads))
        SeqIO.SffIO._check_eof(handle, index_offset, index_length)

    def get(self, offset):
        """Return the SeqRecord starting at the given offset."""
        handle = self._handle
        handle.seek(offset)
        return SeqIO.SffIO._sff_read_seq_record(handle, self._flows_per_read, self._flow_chars, self._key_sequence)

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        handle.seek(offset)
        return SeqIO.SffIO._sff_read_raw_record(handle, self._flows_per_read)