from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
class DifferentialCutsite:
    """Differential enzyme cutsite in an alignment.

    A differential cutsite is a location in an alignment where an enzyme cuts
    at least one sequence and also cannot cut at least one other sequence.

    Members:
     - start - Where it lives in the alignment.
     - enzyme - The enzyme that causes this.
     - cuts_in - A list of sequences (as indexes into the alignment) the
       enzyme cuts in.
     - blocked_in - A list of sequences (as indexes into the alignment) the
       enzyme is blocked in.

    """

    def __init__(self, **kwds):
        """Initialize a DifferentialCutsite.

        Each member (as listed in the class description) should be included as a
        keyword.
        """
        self.start = int(kwds['start'])
        self.enzyme = kwds['enzyme']
        self.cuts_in = kwds['cuts_in']
        self.blocked_in = kwds['blocked_in']