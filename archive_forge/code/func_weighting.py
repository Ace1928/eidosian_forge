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
def weighting(self):
    """Calculate the Phylogenetic weight matrix.

        Constructed from the logarithmic transformation of the
        transformation matrix.
        """
    for k in self.data:
        if self.data[k] != 0:
            self.data[k] = -math.log(self.data[k])
    return self