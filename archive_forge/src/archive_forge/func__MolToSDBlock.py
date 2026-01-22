import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _MolToSDBlock(mol):
    mol_block = Chem.MolToMolBlock(mol, kekulize=False)
    tag_data = []
    for name in mol.GetPropNames():
        if name.startswith('_'):
            continue
        value = mol.GetProp(name)
        tag_data.append('> <' + name + '>\n')
        tag_data.append(value + '\n')
        tag_data.append('\n')
    tag_data.append('$$$$\n')
    return mol_block + ''.join(tag_data)