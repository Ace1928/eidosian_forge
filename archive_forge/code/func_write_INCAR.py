from __future__ import annotations
import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any
import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def write_INCAR(self, incar_input: str='INCAR', incar_output: str='INCAR.lobster', poscar_input: str='POSCAR', isym: int=-1, further_settings: dict | None=None):
    """Will only make the run static, insert nbands, make ISYM=-1, set LWAVE=True and write a new INCAR.
        You have to check for the rest.

        Args:
            incar_input (str): path to input INCAR
            incar_output (str): path to output INCAR
            poscar_input (str): path to input POSCAR
            isym (int): isym equal to -1 or 0 are possible. Current Lobster version only allow -1.
            further_settings (dict): A dict can be used to include further settings, e.g. {"ISMEAR":-5}
        """
    incar = Incar.from_file(incar_input)
    warnings.warn('Please check your incar_input before using it. This method only changes three settings!')
    if isym == -1:
        incar['ISYM'] = -1
    elif isym == 0:
        incar['ISYM'] = 0
    else:
        raise ValueError(f'Got isym={isym!r}, must be -1 or 0')
    incar['NSW'] = 0
    incar['LWAVE'] = True
    incar['NBANDS'] = self._get_nbands(Structure.from_file(poscar_input))
    if further_settings is not None:
        for key, item in further_settings.items():
            incar[key] = item
    incar.write_file(incar_output)