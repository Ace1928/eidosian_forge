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
def _name_n_vector(self, opts, separator='='):
    """Extract name and check that it's not in vector format (PRIVATE)."""
    rest = opts.rest()
    name = opts.next_word()
    if name == '*':
        name = opts.next_word()
    if not name:
        raise NexusError(f'Formatting error in line: {rest} ')
    name = quotestrip(name)
    if opts.peek_nonwhitespace == '(':
        open = opts.next_nonwhitespace()
        qualifier = open.next_word()
        close = opts.next_nonwhitespace()
        if qualifier.lower() == 'vector':
            raise NexusError(f'Unsupported VECTOR format in line {opts}')
        elif qualifier.lower() != 'standard':
            raise NexusError(f'Unknown qualifier {qualifier} in line {opts}')
    if opts.next_nonwhitespace() != separator:
        raise NexusError(f'Formatting error in line: {rest} ')
    return name