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
class MPELNESSet(MPEELSDictSet):
    """FeffDictSet for ELNES spectroscopy."""
    CONFIG = loadfn(f'{MODULE_DIR}/MPELNESSet.yaml')

    def __init__(self, absorbing_atom, structure, edge: str='K', radius: float=10.0, beam_energy: float=100, beam_direction=None, collection_angle: float=1, convergence_angle: float=1, user_eels_settings=None, nkpts: int=1000, user_tag_settings: dict | None=None, **kwargs):
        """
        Args:
            absorbing_atom (str/int): absorbing atom symbol or site index
            structure (Structure): input structure
            edge (str): absorption edge
            radius (float): cluster radius in Angstroms.
            beam_energy (float): Incident beam energy in keV
            beam_direction (list): Incident beam direction. If None, the
                cross section will be averaged.
            collection_angle (float): Detector collection angle in mrad.
            convergence_angle (float): Beam convergence angle in mrad.
            user_eels_settings (dict): override default EELS config.
                See MPELNESSet.yaml for supported keys.
            nkpts (int): Total number of kpoints in the brillouin zone. Used
                only when feff is run in the reciprocal space mode.
            user_tag_settings (dict): override default tag settings
            **kwargs: Passthrough to FEFFDictSet.
        """
        super().__init__(absorbing_atom, structure, edge, 'ELNES', radius, beam_energy, beam_direction, collection_angle, convergence_angle, MPELNESSet.CONFIG, user_eels_settings=user_eels_settings, nkpts=nkpts, user_tag_settings=user_tag_settings, **kwargs)