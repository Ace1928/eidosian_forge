import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _sff_file_header(handle):
    """Read in an SFF file header (PRIVATE).

    Assumes the handle is at the start of the file, will read forwards
    though the header and leave the handle pointing at the first record.
    Returns a tuple of values from the header (header_length, index_offset,
    index_length, number_of_reads, flows_per_read, flow_chars, key_sequence)

    >>> with open("Roche/greek.sff", "rb") as handle:
    ...     values = _sff_file_header(handle)
    ...
    >>> print(values[0])
    840
    >>> print(values[1])
    65040
    >>> print(values[2])
    256
    >>> print(values[3])
    24
    >>> print(values[4])
    800
    >>> values[-1]
    'TCAG'

    """
    fmt = '>4s4BQIIHHHB'
    assert 31 == struct.calcsize(fmt)
    data = handle.read(31)
    if not data:
        raise ValueError('Empty file.')
    elif len(data) < 31:
        raise ValueError('File too small to hold a valid SFF header.')
    try:
        magic_number, ver0, ver1, ver2, ver3, index_offset, index_length, number_of_reads, header_length, key_length, number_of_flows_per_read, flowgram_format = struct.unpack(fmt, data)
    except TypeError:
        raise StreamModeError('SFF files must be opened in binary mode.') from None
    if magic_number in [_hsh, _srt, _mft]:
        raise ValueError('Handle seems to be at SFF index block, not start')
    if magic_number != _sff:
        raise ValueError(f"SFF file did not start '.sff', but {magic_number!r}")
    if (ver0, ver1, ver2, ver3) != (0, 0, 0, 1):
        raise ValueError('Unsupported SFF version in header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
    if flowgram_format != 1:
        raise ValueError('Flowgram format code %i not supported' % flowgram_format)
    if (index_offset != 0) ^ (index_length != 0):
        raise ValueError('Index offset %i but index length %i' % (index_offset, index_length))
    flow_chars = handle.read(number_of_flows_per_read).decode('ASCII')
    key_sequence = handle.read(key_length).decode('ASCII')
    assert header_length % 8 == 0
    padding = header_length - number_of_flows_per_read - key_length - 31
    assert 0 <= padding < 8, padding
    if handle.read(padding).count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post header %i byte null padding region contained data.' % padding, BiopythonParserWarning)
    return (header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence)