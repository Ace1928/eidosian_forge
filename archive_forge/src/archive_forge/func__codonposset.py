from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _codonposset(self, options):
    """Read codon positions from a codons block as written from McClade (PRIVATE).

        Here codonposset is just a fancy name for a character partition with
        the name CodonPositions and the partitions N,1,2,3
        """
    prev_partitions = list(self.charpartitions.keys())
    self._charpartition(options)
    codonname = [n for n in self.charpartitions if n not in prev_partitions]
    if codonname == [] or len(codonname) > 1:
        raise NexusError(f'Formatting Error in codonposset: {options} ')
    else:
        self.codonposset = codonname[0]