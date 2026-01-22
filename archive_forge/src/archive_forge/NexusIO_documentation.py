from typing import IO, Iterator, Optional
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentWriter
from Bio.Nexus import Nexus
from Bio.SeqRecord import SeqRecord
Return 'protein', 'dna', or 'rna' based on records' molecule type (PRIVATE).

        All the records must have a molecule_type annotation, and they must
        agree.

        Raises an exception if this is not possible.
        