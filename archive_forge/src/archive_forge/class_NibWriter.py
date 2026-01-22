import binascii
import struct
import sys
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class NibWriter(SequenceWriter):
    """Nib file writer."""

    def __init__(self, target):
        """Initialize a Nib writer object.

        Arguments:
         - target - output stream opened in binary mode, or a path to a file

        """
        super().__init__(target, mode='wb')

    def write_header(self):
        """Write the file header."""
        super().write_header()
        handle = self.handle
        byteorder = sys.byteorder
        if byteorder == 'little':
            signature = '3a3de96b'
        elif byteorder == 'big':
            signature = '6be93d3a'
        else:
            raise RuntimeError(f'unexpected system byte order {byteorder}')
        handle.write(bytes.fromhex(signature))

    def write_record(self, record):
        """Write a single record to the output file."""
        handle = self.handle
        sequence = record.seq
        nucleotides = bytes(sequence)
        length = len(sequence)
        handle.write(struct.pack('i', length))
        table = bytes.maketrans(b'TCAGNtcagn', b'0123489abc')
        padding = length % 2
        suffix = padding * b'T'
        nucleotides += suffix
        if not set(nucleotides).issubset(b'ACGTNacgtn'):
            raise ValueError('Sequence should contain A,C,G,T,N,a,c,g,t,n only')
        indices = nucleotides.translate(table)
        handle.write(binascii.unhexlify(indices))

    def write_file(self, records):
        """Write the complete file with the records, and return the number of records."""
        count = super().write_file(records, mincount=1, maxcount=1)
        return count