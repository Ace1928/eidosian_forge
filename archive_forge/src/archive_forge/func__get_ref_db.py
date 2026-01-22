import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
def _get_ref_db(self):
    """Get function for the database reference property (PRIVATE)."""
    warnings.warn('Please use .location.ref_db rather than .ref_db', BiopythonDeprecationWarning)
    try:
        return self.location.ref_db
    except AttributeError:
        return None