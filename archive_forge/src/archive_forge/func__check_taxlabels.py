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
def _check_taxlabels(self, taxon):
    """Check for presence of taxon in self.taxlabels (PRIVATE)."""
    nextaxa = {t.replace(' ', '_'): t for t in self.taxlabels}
    nexid = taxon.replace(' ', '_')
    return nextaxa.get(nexid)