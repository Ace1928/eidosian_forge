from __future__ import annotations
import re
from math import fabs
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

        Reads an xr-formatted file to create an Xr object.

        Args:
            filename (str): name of file to read from.
            use_cores (bool): use core positions and discard shell
                    positions if set to True (default). Otherwise,
                    use shell positions and discard core positions.
            thresh (float): relative threshold for consistency check
                    between cell parameters (lengths and angles) from
                    header information and cell vectors, respectively.

        Returns:
            xr (Xr): Xr object corresponding to the input
                    file.
        