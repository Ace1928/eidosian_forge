import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
def _get_ref(self):
    """Get function for the reference property (PRIVATE)."""
    warnings.warn('Please use .location.ref rather than .ref', BiopythonDeprecationWarning)
    try:
        return self.location.ref
    except AttributeError:
        return None