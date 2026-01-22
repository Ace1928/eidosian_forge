import warnings
from re import match
from struct import pack
from struct import unpack
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _read_feature(handle, record):
    """Read a single sequence feature."""
    name = _read_pstring(handle)
    desc = _read_pstring(handle)
    type = _read_pstring(handle) or 'misc_feature'
    start = _read_pstring_as_integer(handle)
    end = _read_pstring_as_integer(handle)
    forward, display, arrow = unpack('>BBxB', _read(handle, 4))
    if forward:
        strand = 1
    else:
        strand = -1
        start, end = (end, start)
    _read_pstring(handle)
    location = SimpleLocation(start - 1, end, strand=strand)
    qualifiers = {}
    if name:
        qualifiers['label'] = [name]
    _parse_feature_description(desc, qualifiers)
    feature = SeqFeature(location, type=type, qualifiers=qualifiers)
    record.features.append(feature)