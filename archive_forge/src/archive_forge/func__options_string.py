from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def _options_string(self):
    """Return the option string."""
    if len(self.options) > 0:
        opt_str = ''
        for op in self.options:
            if self._sized_op:
                opt_str += f'{op[0]}={op[1]} '
            else:
                opt_str += f'{op} '
        return opt_str.strip()
    return ''