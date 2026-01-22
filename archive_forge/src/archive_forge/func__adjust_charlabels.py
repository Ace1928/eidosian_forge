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
def _adjust_charlabels(self, exclude=None, insert=None):
    """Return adjusted indices of self.charlabels if characters are excluded or inserted (PRIVATE)."""
    if exclude and insert:
        raise NexusError("Can't exclude and insert at the same time")
    if not self.charlabels:
        return None
    labels = sorted(self.charlabels)
    newcharlabels = {}
    if exclude:
        exclude.sort()
        exclude.append(sys.maxsize)
        excount = 0
        for c in labels:
            if c not in exclude:
                while c > exclude[excount]:
                    excount += 1
                newcharlabels[c - excount] = self.charlabels[c]
    elif insert:
        insert.sort()
        insert.append(sys.maxsize)
        icount = 0
        for c in labels:
            while c >= insert[icount]:
                icount += 1
            newcharlabels[c + icount] = self.charlabels[c]
    else:
        return self.charlabels
    return newcharlabels