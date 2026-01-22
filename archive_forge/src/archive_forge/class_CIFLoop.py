import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
class CIFLoop:

    def __init__(self):
        self.names = []
        self.formats = []
        self.arrays = []

    def add(self, name, array, fmt):
        assert name.startswith('_')
        self.names.append(name)
        self.formats.append(fmt)
        self.arrays.append(array)
        if len(self.arrays[0]) != len(self.arrays[-1]):
            raise ValueError(f'Loop data "{name}" has {len(array)} elements, expected {{len(self.arrays[0])}}')

    def tostring(self):
        lines = []
        append = lines.append
        append('loop_')
        for name in self.names:
            append(f'  {name}')
        template = '  ' + '  '.join(self.formats)
        ncolumns = len(self.arrays)
        nrows = len(self.arrays[0]) if ncolumns > 0 else 0
        for row in range(nrows):
            arraydata = [array[row] for array in self.arrays]
            line = template.format(*arraydata)
            append(line)
        append('')
        return '\n'.join(lines)