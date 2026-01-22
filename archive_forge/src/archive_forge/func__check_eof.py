import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _check_eof(handle, index_offset, index_length):
    """Check final padding is OK (8 byte alignment) and file ends (PRIVATE).

    Will attempt to spot apparent SFF file concatenation and give an error.

    Will not attempt to seek, only moves the handle forward.
    """
    offset = handle.tell()
    extra = b''
    padding = 0
    if index_offset and offset <= index_offset:
        if offset < index_offset:
            raise ValueError('Gap of %i bytes after final record end %i, before %i where index starts?' % (index_offset - offset, offset, index_offset))
        handle.read(index_offset + index_length - offset)
        offset = index_offset + index_length
        if offset != handle.tell():
            raise ValueError('Wanted %i, got %i, index is %i to %i' % (offset, handle.tell(), index_offset, index_offset + index_length))
    if offset % 8:
        padding = 8 - offset % 8
        extra = handle.read(padding)
    if padding >= 4 and extra[-4:] == _sff:
        raise ValueError("Your SFF file is invalid, post index %i byte null padding region ended '.sff' which could be the start of a concatenated SFF file? See offset %i" % (padding, offset))
    if padding and (not extra):
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is technically invalid as it is missing a terminal %i byte null padding region.' % padding, BiopythonParserWarning)
        return
    if extra.count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post index %i byte null padding region contained data: %r' % (padding, extra), BiopythonParserWarning)
    offset = handle.tell()
    if offset % 8 != 0:
        raise ValueError('Wanted offset %i %% 8 = %i to be zero' % (offset, offset % 8))
    extra = handle.read(4)
    if extra == _sff:
        raise ValueError('Additional data at end of SFF file, perhaps multiple SFF files concatenated? See offset %i' % offset)
    elif extra:
        raise ValueError('Additional data at end of SFF file, see offset %i' % offset)