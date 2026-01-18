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
def weighted_stepmatrix(self, name='your_name_here', exclude=(), delete=()):
    """Calculate a stepmatrix for weighted parsimony.

        See Wheeler (1990), Cladistics 6:269-275 and
        Felsenstein (1981), Biol. J. Linn. Soc. 16:183-196
        """
    m = StepMatrix(self.unambiguous_letters, self.gap)
    for site in [s for s in range(self.nchar) if s not in exclude]:
        cstatus = self.cstatus(site, delete)
        for i, b1 in enumerate(cstatus[:-1]):
            for b2 in cstatus[i + 1:]:
                m.add(b1.upper(), b2.upper(), 1)
    return m.transformation().weighting().smprint(name=name)