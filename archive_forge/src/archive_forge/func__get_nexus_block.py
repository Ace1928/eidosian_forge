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
def _get_nexus_block(self, file_contents):
    """Return a generator for looping through Nexus blocks (PRIVATE)."""
    inblock = False
    blocklines = []
    while file_contents:
        cl = file_contents.pop(0)
        if cl.lower().startswith('begin'):
            if not inblock:
                inblock = True
                title = cl.split()[1].lower()
            else:
                raise NexusError(f'Illegal block nesting in block {title}')
        elif cl.lower().startswith('end'):
            if inblock:
                inblock = False
                yield (title, blocklines)
                blocklines = []
            else:
                raise NexusError("Unmatched 'end'.")
        elif inblock:
            blocklines.append(cl)