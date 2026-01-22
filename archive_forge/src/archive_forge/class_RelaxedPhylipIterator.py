from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class RelaxedPhylipIterator(PhylipIterator):
    """Relaxed Phylip format Iterator."""

    def _split_id(self, line):
        """Extract the sequence ID from a Phylip line (PRIVATE).

        Returns a tuple containing: (sequence_id, sequence_residues)

        For relaxed format split at the first whitespace character.
        """
        seq_id, sequence = line.split(None, 1)
        sequence = sequence.strip().replace(' ', '')
        return (seq_id, sequence)