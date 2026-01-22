from __future__ import annotations
import re
from io import StringIO
from typing import TYPE_CHECKING, cast
import pandas as pd
from monty.io import zopen
from pymatgen.core import Molecule, Structure
from pymatgen.core.structure import SiteCollection
def _frame_str(self, frame_mol):
    output = [str(len(frame_mol)), frame_mol.formula]
    prec = self.precision
    fmt = f'{{}} {{:.{prec}f}} {{:.{prec}f}} {{:.{prec}f}}'
    for site in frame_mol:
        output.append(fmt.format(site.specie, site.x, site.y, site.z))
    return '\n'.join(output)