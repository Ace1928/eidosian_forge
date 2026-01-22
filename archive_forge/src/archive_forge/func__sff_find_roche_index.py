import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _sff_find_roche_index(handle):
    """Locate any existing Roche style XML meta data and read index (PRIVATE).

    Makes a number of hard coded assumptions based on reverse engineered SFF
    files from Roche 454 machines.

    Returns a tuple of read count, SFF "index" offset and size, XML offset
    and size, and the actual read index offset and size.

    Raises a ValueError for unsupported or non-Roche index blocks.
    """
    handle.seek(0)
    header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence = _sff_file_header(handle)
    assert handle.tell() == header_length
    if not index_offset or not index_length:
        raise ValueError('No index present in this SFF file')
    handle.seek(index_offset)
    fmt = '>4s4B'
    fmt_size = struct.calcsize(fmt)
    data = handle.read(fmt_size)
    if not data:
        raise ValueError('Premature end of file? Expected index of size %i at offset %i, found nothing' % (index_length, index_offset))
    if len(data) < fmt_size:
        raise ValueError('Premature end of file? Expected index of size %i at offset %i, found %r' % (index_length, index_offset, data))
    magic_number, ver0, ver1, ver2, ver3 = struct.unpack(fmt, data)
    if magic_number == _mft:
        if (ver0, ver1, ver2, ver3) != (49, 46, 48, 48):
            raise ValueError('Unsupported version in .mft index header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
        fmt2 = '>LL'
        fmt2_size = struct.calcsize(fmt2)
        xml_size, data_size = struct.unpack(fmt2, handle.read(fmt2_size))
        if index_length != fmt_size + fmt2_size + xml_size + data_size:
            raise ValueError('Problem understanding .mft index header, %i != %i + %i + %i + %i' % (index_length, fmt_size, fmt2_size, xml_size, data_size))
        return (number_of_reads, header_length, index_offset, index_length, index_offset + fmt_size + fmt2_size, xml_size, index_offset + fmt_size + fmt2_size + xml_size, data_size)
    elif magic_number == _srt:
        if (ver0, ver1, ver2, ver3) != (49, 46, 48, 48):
            raise ValueError('Unsupported version in .srt index header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
        data = handle.read(4)
        if data != _null * 4:
            raise ValueError('Did not find expected null four bytes in .srt index')
        return (number_of_reads, header_length, index_offset, index_length, 0, 0, index_offset + fmt_size + 4, index_length - fmt_size - 4)
    elif magic_number == _hsh:
        raise ValueError('Hash table style indexes (.hsh) in SFF files are not (yet) supported')
    else:
        raise ValueError(f'Unknown magic number {magic_number!r} in SFF index header:\n{data!r}')