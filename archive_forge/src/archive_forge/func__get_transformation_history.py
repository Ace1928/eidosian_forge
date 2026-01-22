from __future__ import annotations
import abc
import json
import logging
import os
import warnings
from glob import glob
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.gaussian import GaussianOutput
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Dynmat, Oszicar, Vasprun
def _get_transformation_history(path):
    """Checks for a transformations.json* file and returns the history."""
    trans_json = glob(f'{path}/transformations.json*')
    if trans_json:
        try:
            with zopen(trans_json[0]) as file:
                return json.load(file)['history']
        except Exception:
            return None
    return None