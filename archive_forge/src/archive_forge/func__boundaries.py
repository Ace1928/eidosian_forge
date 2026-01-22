import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
def _boundaries(self, start, end):
    """Set boundaries to correct values (PRIVATE).

        Format the boundaries for use with the methods that limit the
        search to only part of the sequence given to analyse.
        """
    if not isinstance(start, int):
        raise TypeError(f'expected int, got {type(start)} instead')
    if not isinstance(end, int):
        raise TypeError(f'expected int, got {type(end)} instead')
    if start < 1:
        start += len(self.sequence)
    if end < 1:
        end += len(self.sequence)
    if start < end:
        pass
    else:
        start, end = (end, start)
    if start < end:
        return (start, end, self._test_normal)