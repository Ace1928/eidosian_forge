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
def write_lobsterin(self, path='lobsterin', overwritedict=None):
    """
        Writes a lobsterin file.

        Args:
            path (str): filename of the lobsterin file that will be written
            overwritedict (dict): dict that can be used to overwrite lobsterin, e.g. {"skipdos": True}
        """
    if overwritedict is not None:
        for key, entry in overwritedict.items():
            found = False
            for key2 in self:
                if key.lower() == key2.lower():
                    self[key2] = entry
                    found = True
            if not found:
                self[key] = entry
    filename = path
    with open(filename, mode='w', encoding='utf-8') as file:
        for key in Lobsterin.AVAILABLE_KEYWORDS:
            if key.lower() in [element.lower() for element in self]:
                if key.lower() in [element.lower() for element in Lobsterin.FLOAT_KEYWORDS]:
                    file.write(f'{key} {self.get(key)}\n')
                elif key.lower() in [element.lower() for element in Lobsterin.BOOLEAN_KEYWORDS]:
                    for key_here in self:
                        if key.lower() == key_here.lower():
                            new_key = key_here
                    if self.get(new_key):
                        file.write(key + '\n')
                elif key.lower() in [element.lower() for element in Lobsterin.STRING_KEYWORDS]:
                    file.write(f'{key} {self.get(key)}\n')
                elif key.lower() in [element.lower() for element in Lobsterin.LISTKEYWORDS]:
                    for entry in self.get(key):
                        file.write(f'{key} {entry}\n')