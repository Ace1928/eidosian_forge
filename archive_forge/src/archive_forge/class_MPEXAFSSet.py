from __future__ import annotations
import abc
import logging
import os
import sys
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.feff.inputs import Atoms, Header, Potential, Tags
class MPEXAFSSet(FEFFDictSet):
    """FeffDictSet for EXAFS spectroscopy."""
    CONFIG = loadfn(f'{MODULE_DIR}/MPEXAFSSet.yaml')

    def __init__(self, absorbing_atom, structure, edge: str='K', radius: float=10.0, nkpts: int=1000, user_tag_settings: dict | None=None, **kwargs):
        """
        Args:
            absorbing_atom (str/int): absorbing atom symbol or site index
            structure (Structure): input structure
            edge (str): absorption edge
            radius (float): cluster radius in Angstroms.
            nkpts (int): Total number of kpoints in the brillouin zone. Used
                only when feff is run in the reciprocal space mode.
            user_tag_settings (dict): override default tag settings
            **kwargs: Passthrough to FEFFDictSet.
        """
        super().__init__(absorbing_atom, structure, radius, MPEXAFSSet.CONFIG, edge=edge, spectrum='EXAFS', nkpts=nkpts, user_tag_settings=user_tag_settings, **kwargs)