from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
class DatabaseReport:
    """Holds information about a database report.

    Members:
    database_name              List of database names.  (can have multiple dbs)
    num_letters_in_database    Number of letters in the database.  (int)
    num_sequences_in_database  List of number of sequences in the database.
    posted_date                List of the dates the databases were posted.
    ka_params                  A tuple of (lambda, k, h) values.  (floats)
    gapped                     # XXX this isn't set right!
    ka_params_gap              A tuple of (lambda, k, h) values.  (floats)

    """

    def __init__(self):
        """Initialize the class."""
        self.database_name = []
        self.posted_date = []
        self.num_letters_in_database = []
        self.num_sequences_in_database = []
        self.ka_params = (None, None, None)
        self.gapped = 0
        self.ka_params_gap = (None, None, None)