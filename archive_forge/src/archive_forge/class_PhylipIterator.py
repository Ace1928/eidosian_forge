from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class PhylipIterator(AlignmentIterator):
    """Reads a Phylip alignment file returning a MultipleSeqAlignment iterator.

    Record identifiers are limited to at most 10 characters.

    It only copes with interlaced phylip files!  Sequential files won't work
    where the sequences are split over multiple lines.

    For more information on the file format, please see:
    http://evolution.genetics.washington.edu/phylip/doc/sequence.html
    http://evolution.genetics.washington.edu/phylip/doc/main.html#inputfiles
    """
    id_width = _PHYLIP_ID_WIDTH
    _header = None

    def _is_header(self, line):
        line = line.strip()
        parts = [x for x in line.split() if x]
        if len(parts) != 2:
            return False
        try:
            number_of_seqs = int(parts[0])
            length_of_seqs = int(parts[1])
            return True
        except ValueError:
            return False

    def _split_id(self, line):
        """Extract the sequence ID from a Phylip line (PRIVATE).

        Returning a tuple containing: (sequence_id, sequence_residues)

        The first 10 characters in the line are are the sequence id, the
        remainder are sequence data.
        """
        seq_id = line[:self.id_width].strip()
        seq = line[self.id_width:].strip().replace(' ', '')
        return (seq_id, seq)

    def __next__(self):
        """Parse the next alignment from the handle."""
        handle = self.handle
        if self._header is None:
            line = handle.readline()
        else:
            line = self._header
            self._header = None
        if not line:
            raise StopIteration
        line = line.strip()
        parts = [x for x in line.split() if x]
        if len(parts) != 2:
            raise ValueError('First line should have two integers')
        try:
            number_of_seqs = int(parts[0])
            length_of_seqs = int(parts[1])
        except ValueError:
            raise ValueError('First line should have two integers') from None
        assert self._is_header(line)
        if self.records_per_alignment is not None and self.records_per_alignment != number_of_seqs:
            raise ValueError('Found %i records in this alignment, told to expect %i' % (number_of_seqs, self.records_per_alignment))
        ids = []
        seqs = []
        for i in range(number_of_seqs):
            line = handle.readline().rstrip()
            sequence_id, s = self._split_id(line)
            ids.append(sequence_id)
            if '.' in s:
                raise ValueError(_NO_DOTS)
            seqs.append([s])
        line = ''
        while True:
            while '' == line.strip():
                line = handle.readline()
                if not line:
                    break
            if not line:
                break
            if self._is_header(line):
                self._header = line
                break
            for i in range(number_of_seqs):
                s = line.strip().replace(' ', '')
                if '.' in s:
                    raise ValueError(_NO_DOTS)
                seqs[i].append(s)
                line = handle.readline()
                if not line and i + 1 < number_of_seqs:
                    raise ValueError('End of file mid-block')
            if not line:
                break
        records = (SeqRecord(Seq(''.join(s)), id=i, name=i, description=i) for i, s in zip(ids, seqs))
        return MultipleSeqAlignment(records)