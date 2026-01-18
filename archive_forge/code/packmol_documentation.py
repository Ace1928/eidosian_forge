from __future__ import annotations
import os
import subprocess
from pathlib import Path
from shutil import which
import numpy as np
from pymatgen.core import Molecule
from pymatgen.io.core import InputGenerator, InputSet

        Generate a Packmol InputSet for a set of molecules.

        Args:
            molecules: A list of dict containing information about molecules to pack
                into the box. Each dict requires three keys:
                    1. "name" - the structure name
                    2. "number" - the number of that molecule to pack into the box
                    3. "coords" - Coordinates in the form of either a Molecule object or
                        a path to a file.

        Example:
                    {"name": "water",
                     "number": 500,
                     "coords": "/path/to/input/file.xyz"}
            box: A list of box dimensions xlo, ylo, zlo, xhi, yhi, zhi, in Å. If set to None
                (default), pymatgen will estimate the required box size based on the volumes of
                the provided molecules.
        