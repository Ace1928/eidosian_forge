from io import StringIO
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.SeqRecord import SeqRecord
from Bio.Nexus import Nexus
Return 'protein', 'dna', or 'rna' based on records' molecule type (PRIVATE).

        All the records must have a molecule_type annotation, and they must
        agree.

        Raises an exception if this is not possible.
        