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
@staticmethod
def _get_potcar_symbols(POTCAR_input: str) -> list:
    """
        Will return the name of the species in the POTCAR.

        Args:
            POTCAR_input (str): string to potcar file

        Returns:
            list of the names of the species in string format
        """
    potcar = Potcar.from_file(POTCAR_input)
    for pot in potcar:
        if pot.potential_type != 'PAW':
            raise OSError('Lobster only works with PAW! Use different POTCARs')
    with zopen(POTCAR_input, mode='r') as file:
        data = file.read()
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    if 'SHA256' in data or 'COPYR' in data:
        warnings.warn('These POTCARs are not compatible with Lobster up to version 4.1.0.\n The keywords SHA256 and COPYR cannot be handled by Lobster \n and will lead to wrong results.')
    if potcar.functional != 'PBE':
        raise OSError('We only have BASIS options for PBE so far')
    return [name['symbol'] for name in potcar.spec]