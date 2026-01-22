from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
class Blast(Header, DatabaseReport, Parameters):
    """Saves the results from a blast search.

    Members:
    descriptions        A list of Description objects.
    alignments          A list of Alignment objects.
    multiple_alignment  A MultipleAlignment object.
    + members inherited from base classes

    """

    def __init__(self):
        """Initialize the class."""
        Header.__init__(self)
        DatabaseReport.__init__(self)
        Parameters.__init__(self)
        self.descriptions = []
        self.alignments = []
        self.multiple_alignment = None