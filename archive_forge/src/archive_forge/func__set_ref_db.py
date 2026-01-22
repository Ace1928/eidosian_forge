import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
def _set_ref_db(self, value):
    """Set function for the database reference property (PRIVATE)."""
    warnings.warn('Please use .location.ref_db rather than .ref_db', BiopythonDeprecationWarning)
    self.location.ref_db = value