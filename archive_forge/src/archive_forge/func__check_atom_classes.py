import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _check_atom_classes(molno, num_atoms, atom_classes):
    if num_atoms != len(atom_classes):
        raise ValueError('mols[%d]: len(atom_classes) must be the same as the number of atoms' % (molno,))
    for atom_class in atom_classes:
        if not isinstance(atom_class, int):
            raise ValueError('mols[%d]: atom_class elements must be integers' % (molno,))
        if not 1 <= atom_class < 1000:
            raise ValueError('mols[%d]: atom_class elements must be in the range 1 <= value < 1000' % (molno,))