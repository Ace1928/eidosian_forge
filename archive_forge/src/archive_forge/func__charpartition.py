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
def _charpartition(self, options):
    """Collect character partition from NEXUS file (PRIVATE)."""
    charpartition = {}
    quotelevel = False
    opts = CharBuffer(options)
    name = self._name_n_vector(opts)
    if not name:
        raise NexusError(f'Formatting error in charpartition: {options} ')
    sub = ''
    while True:
        w = next(opts)
        if w is None or (w == ',' and (not quotelevel)):
            subname, subindices = self._get_indices(sub, set_type=CHARSET, separator=':')
            charpartition[subname] = _make_unique(subindices)
            sub = ''
            if w is None:
                break
        else:
            if w == "'":
                quotelevel = not quotelevel
            sub += w
    self.charpartitions[name] = charpartition