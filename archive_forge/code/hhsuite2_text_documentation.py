import re
import warnings
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
Return True if match is not a Consensus column (PRIVATE).

            It's not possible to distinguish a sequence line from a Consensus line with
            a regexp, so need to check the ID column.
            