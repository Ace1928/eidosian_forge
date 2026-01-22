from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def V_xx(self):
    """Returns: First diagonal element."""
    diags = np.diag(self.principal_axis_system)
    return sorted(diags, key=np.abs)[0]