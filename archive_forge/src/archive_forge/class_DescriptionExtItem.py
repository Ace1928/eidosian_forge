from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
class DescriptionExtItem:
    """Stores information about one record in hit description for BLASTXML version 2.

    Members:
    id              Database identifier
    title           Title of the hit.
    """

    def __init__(self):
        """Initialize the class."""
        self.id = None
        self.title = None
        self.accession = None
        self.taxid = None
        self.sciname = None

    def __str__(self):
        """Return the description identifier and title as a string."""
        return f'{self.id} {self.title}'