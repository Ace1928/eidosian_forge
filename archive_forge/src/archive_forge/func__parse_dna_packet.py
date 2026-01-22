from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _parse_dna_packet(length, data, record):
    """Parse a DNA sequence packet.

    A DNA sequence packet contains a single byte flag followed by the
    sequence itself.
    """
    if record.seq:
        raise ValueError('The file contains more than one DNA packet')
    flags, sequence = unpack('>B%ds' % (length - 1), data)
    record.seq = Seq(sequence.decode('ASCII'))
    record.annotations['molecule_type'] = 'DNA'
    if flags & 1:
        record.annotations['topology'] = 'circular'
    else:
        record.annotations['topology'] = 'linear'