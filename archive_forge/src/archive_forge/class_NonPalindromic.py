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
class NonPalindromic(AbstractCut):
    """Implement methods for enzymes with non-palindromic recognition sites.

    Palindromic means : the recognition site and its reverse complement are
                        identical.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _search(cls):
        """Return a list of cutting sites of the enzyme in the sequence (PRIVATE).

        For internal use only.

        Implement the search method for non palindromic enzymes.
        """
        iterator = cls.dna.finditer(cls.compsite, cls.size)
        cls.results = []
        modif = cls._modify
        revmodif = cls._rev_modify
        s = str(cls)
        cls.on_minus = []
        for start, group in iterator:
            if group(s):
                cls.results += list(modif(start))
            else:
                cls.on_minus += list(revmodif(start))
        cls.results += cls.on_minus
        if cls.results:
            cls.results.sort()
            cls._drop()
        return cls.results

    @classmethod
    def is_palindromic(cls):
        """Return if the enzyme has a palindromic recoginition site."""
        return False