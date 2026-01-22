import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell
def _read_construct_regex(lines):
    """
    Utility for constructing  regular expressions used by reader.
    """
    lines = [l.strip() for l in lines]
    lines_re = '|'.join(lines)
    lines_re = lines_re.replace(' ', '\\s+')
    lines_re = lines_re.replace('(', '\\(')
    lines_re = lines_re.replace(')', '\\)')
    return '({})'.format(lines_re)