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
@classmethod
def _modify(cls, location):
    """Return a generator that moves the cutting position by 1 (PRIVATE).

        For internal use only.

        location is an integer corresponding to the location of the match for
        the enzyme pattern in the sequence.
        _modify returns the real place where the enzyme will cut.

        example::

            EcoRI pattern : GAATTC
            EcoRI will cut after the G.
            so in the sequence:
                     ______
            GAATACACGGAATTCGA
                     |
                     10
            dna.finditer(GAATTC, 6) will return 10 as G is the 10th base
            EcoRI cut after the G so:
            EcoRI._modify(10) -> 11.

        if the enzyme cut twice _modify will returns two integer corresponding
        to each cutting site.
        """
    yield (location + cls.fst5)
    yield (location + cls.scd5)