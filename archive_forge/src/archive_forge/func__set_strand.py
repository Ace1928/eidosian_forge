import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
def _set_strand(self, value):
    """Set function for the strand property (PRIVATE)."""
    for loc in self.parts:
        loc.strand = value