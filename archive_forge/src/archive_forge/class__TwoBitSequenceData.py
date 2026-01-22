from Bio.Seq import Seq
from Bio.Seq import SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord
from . import _twoBitIO  # type: ignore
from .Interfaces import SequenceIterator
class _TwoBitSequenceData(SequenceDataAbstractBaseClass):
    """Stores information needed to retrieve sequence data from a .2bit file (PRIVATE).

    Objects of this class store the file position at which the sequence data
    start, the sequence length, and the start and end position of unknown (N)
    and masked (lowercase) letters in the sequence.

    Only two methods are provided: __len__ and __getitem__. The former will
    return the length of the sequence, while the latter returns the sequence
    (as a bytes object) for the requested region. The full sequence of a record
    is loaded only if explicitly requested.
    """
    __slots__ = ('stream', 'offset', 'length', 'nBlocks', 'maskBlocks')

    def __init__(self, stream, offset, length):
        """Initialize the file stream and file position of the sequence data."""
        self.stream = stream
        self.offset = offset
        self.length = length
        super().__init__()

    def __getitem__(self, key):
        """Return the sequence contents (as a bytes object) for the requested region."""
        length = self.length
        if isinstance(key, slice):
            start, end, step = key.indices(length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
        else:
            if key < 0:
                key += length
                if key < 0:
                    raise IndexError('index out of range')
            start = key
            end = key + 1
            step = 1
            size = 1
        byteStart = start // 4
        byteEnd = (end + 3) // 4
        byteSize = byteEnd - byteStart
        stream = self.stream
        try:
            stream.seek(self.offset + byteStart)
        except ValueError as exception:
            if str(exception) == 'seek of closed file':
                raise ValueError('cannot retrieve sequence: file is closed') from None
            raise
        data = np.fromfile(stream, dtype='uint8', count=byteSize)
        sequence = _twoBitIO.convert(data, start, end, step, self.nBlocks, self.maskBlocks)
        if isinstance(key, slice):
            return sequence
        else:
            return ord(sequence)

    def __len__(self):
        """Get the sequence length."""
        return self.length

    def upper(self):
        """Remove the sequence mask."""
        data = _TwoBitSequenceData(self.stream, self.offset, self.length)
        data.nBlocks = self.nBlocks[:, :]
        data.maskBlocks = np.empty((0, 2), dtype='uint32')
        return data

    def lower(self):
        """Extend the sequence mask to the full sequence."""
        data = _TwoBitSequenceData(self.stream, self.offset, self.length)
        data.nBlocks = self.nBlocks[:, :]
        data.maskBlocks = np.array([[0, self.length]], dtype='uint32')
        return data