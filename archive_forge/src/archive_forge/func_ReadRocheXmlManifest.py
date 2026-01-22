import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def ReadRocheXmlManifest(handle):
    """Read any Roche style XML manifest data in the SFF "index".

    The SFF file format allows for multiple different index blocks, and Roche
    took advantage of this to define their own index block which also embeds
    an XML manifest string. This is not a publicly documented extension to
    the SFF file format, this was reverse engineered.

    The handle should be to an SFF file opened in binary mode. This function
    will use the handle seek/tell functions and leave the handle in an
    arbitrary location.

    Any XML manifest found is returned as a Python string, which you can then
    parse as appropriate, or reuse when writing out SFF files with the
    SffWriter class.

    Returns a string, or raises a ValueError if an Roche manifest could not be
    found.
    """
    number_of_reads, header_length, index_offset, index_length, xml_offset, xml_size, read_index_offset, read_index_size = _sff_find_roche_index(handle)
    if not xml_offset or not xml_size:
        raise ValueError('No XML manifest found')
    handle.seek(xml_offset)
    return handle.read(xml_size).decode()