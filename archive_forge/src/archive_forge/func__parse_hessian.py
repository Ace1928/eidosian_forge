from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def _parse_hessian(self, file, structure):
    """
        Parse the hessian matrix in the output file.

        Args:
            file: file object
            structure: structure in the output file
        """
    ndf = 3 * len(structure)
    self.hessian = np.zeros((ndf, ndf))
    j_indices = range(5)
    ndf_idx = 0
    while ndf_idx < ndf:
        for i in range(ndf_idx, ndf):
            line = file.readline()
            vals = re.findall('\\s*([+-]?\\d+\\.\\d+[eEdD]?[+-]\\d+)', line)
            vals = [float(val.replace('D', 'E')) for val in vals]
            for val_idx, val in enumerate(vals):
                j = j_indices[val_idx]
                self.hessian[i, j] = val
                self.hessian[j, i] = val
        ndf_idx += len(vals)
        line = file.readline()
        j_indices = [j + 5 for j in j_indices]