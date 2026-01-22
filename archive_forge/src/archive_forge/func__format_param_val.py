from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
@staticmethod
def _format_param_val(param_val) -> str:
    """
        Internal method to format values in the packmol parameter dictionaries.

        Args:
            param_val:
                Some object to turn into String

        Returns:
            String representation of the object
        """
    if isinstance(param_val, list):
        return ' '.join((str(x) for x in param_val))
    return str(param_val)