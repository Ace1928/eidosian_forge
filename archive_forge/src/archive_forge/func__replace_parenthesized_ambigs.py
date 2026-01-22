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
def _replace_parenthesized_ambigs(seq, rev_ambig_values):
    """Replace ambigs in xxx(ACG)xxx format by IUPAC ambiguity code (PRIVATE)."""
    opening = seq.find('(')
    while opening > -1:
        closing = seq.find(')')
        if closing < 0:
            raise NexusError('Missing closing parenthesis in: ' + seq)
        elif closing < opening:
            raise NexusError('Missing opening parenthesis in: ' + seq)
        ambig = ''.join(sorted(seq[opening + 1:closing]))
        ambig_code = rev_ambig_values[ambig.upper()]
        if ambig != ambig.upper():
            ambig_code = ambig_code.lower()
        seq = seq[:opening] + ambig_code + seq[closing + 1:]
        opening = seq.find('(')
    return seq