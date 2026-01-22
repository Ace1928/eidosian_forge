from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
class CifBlock:
    """
    Object for storing cif data. All data is stored in a single dictionary.
    Data inside loops are stored in lists in the data dictionary, and
    information on which keys are grouped together are stored in the loops
    attribute.
    """
    max_len = 70

    def __init__(self, data, loops, header):
        """
        Args:
            data: dict of data to go into the cif. Values should be convertible to string,
                or lists of these if the key is in a loop
            loops: list of lists of keys, grouped by which loop they should appear in
            header: name of the block (appears after the data_ on the first line).
        """
        self.loops = loops
        self.data = data
        self.header = header[:74]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CifBlock):
            return NotImplemented
        return self.loops == other.loops and self.data == other.data and (self.header == other.header)

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self) -> str:
        """Returns the cif string for the data block."""
        out = [f'data_{self.header}']
        keys = list(self.data)
        written = []
        for key in keys:
            if key in written:
                continue
            for loop in self.loops:
                if key in loop:
                    out.append(self._loop_to_str(loop))
                    written.extend(loop)
                    break
            if key not in written:
                v = self._format_field(self.data[key])
                if len(key) + len(v) + 3 < self.max_len:
                    out.append(f'{key}   {v}')
                else:
                    out.extend([key, v])
        return '\n'.join(out)

    def _loop_to_str(self, loop):
        out = 'loop_'
        for line in loop:
            out += '\n ' + line
        for fields in zip(*(self.data[k] for k in loop)):
            line = '\n'
            for val in map(self._format_field, fields):
                if val[0] == ';':
                    out += line + '\n' + val
                    line = '\n'
                elif len(line) + len(val) + 2 < self.max_len:
                    line += '  ' + val
                else:
                    out += line
                    line = '\n  ' + val
            out += line
        return out

    def _format_field(self, val) -> str:
        val = str(val).strip()
        if len(val) > self.max_len:
            return f';\n{textwrap.fill(val, self.max_len)}\n;'
        if val == '':
            return '""'
        if (' ' in val or val[0] == '_') and (not (val[0] == "'" and val[-1] == "'")) and (not (val[0] == '"' and val[-1] == '"')):
            quote = '"' if "'" in val else "'"
            val = quote + val + quote
        return val

    @classmethod
    def _process_string(cls, string):
        string = re.sub('(\\s|^)#.*$', '', string, flags=re.MULTILINE)
        string = re.sub('^\\s*\\n', '', string, flags=re.MULTILINE)
        string = string.encode('ascii', 'ignore').decode('ascii')
        deq = deque()
        multiline = False
        ml = []
        pattern = re.compile('([^\'"\\s][\\S]*)|\'(.*?)\'(?!\\S)|"(.*?)"(?!\\S)')
        for line in string.splitlines():
            if multiline:
                if line.startswith(';'):
                    multiline = False
                    deq.append(('', '', '', ' '.join(ml)))
                    ml = []
                    line = line[1:].strip()
                else:
                    ml.append(line)
                    continue
            if line.startswith(';'):
                multiline = True
                ml.append(line[1:].strip())
            else:
                for string in pattern.findall(line):
                    deq.append(tuple(string))
        return deq

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        Reads CifBlock from string.

        Args:
            string: String representation.

        Returns:
            CifBlock
        """
        deq = cls._process_string(string)
        header = deq.popleft()[0][5:]
        data: dict = {}
        loops = []
        while deq:
            s = deq.popleft()
            if s[0] == '_eof':
                break
            if s[0].startswith('_'):
                try:
                    data[s[0]] = ''.join(deq.popleft())
                except IndexError:
                    data[s[0]] = ''
            elif s[0].startswith('loop_'):
                columns = []
                items = []
                while deq:
                    s = deq[0]
                    if s[0].startswith('loop_') or not s[0].startswith('_'):
                        break
                    columns.append(''.join(deq.popleft()))
                    data[columns[-1]] = []
                while deq:
                    s = deq[0]
                    if s[0].startswith(('loop_', '_')):
                        break
                    items.append(''.join(deq.popleft()))
                n = len(items) // len(columns)
                assert len(items) % n == 0
                loops.append(columns)
                for k, v in zip(columns * n, items):
                    data[k].append(v.strip())
            elif (issue := ''.join(s).strip()):
                warnings.warn(f'Possible issue in CIF file at line: {issue}')
        return cls(data, loops, header)