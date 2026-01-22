from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@dataclass
class MPAbsorptionSet(DictSet):
    """
    MP input set for generating frequency dependent dielectrics.

    Two modes are supported: "IPA" or "RPA".
    A typical sequence is mode="STATIC" -> mode="IPA" -> mode="RPA"(optional)
    For all steps other than the first one (static), the
    recommendation is to use from_prev_calculation on the preceding run in
    the series. It is important to ensure Gamma centred kpoints for the RPA step.

    Args:
        structure (Structure): Input structure.
        mode (str): Supported modes are "IPA", "RPA"
        copy_wavecar (bool): Whether to copy the WAVECAR from a previous run.
            Defaults to True.
        nbands (int): For subsequent calculations, it is generally
            recommended to perform NBANDS convergence starting from the
            NBANDS of the previous run for DIAG, and to use the exact same
            NBANDS for RPA. This parameter is used by
            from_previous_calculation to set nband.
        nbands_factor (int): Multiplicative factor for NBANDS when starting
            from a previous calculation. Only applies if mode=="IPA".
            Need to be tested for convergence.
        reciprocal_density: the k-points density
        nkred: the reduced number of kpoints to calculate, equal to the k-mesh.
            Only applies in "RPA" mode because of the q->0 limit.
        nedos: the density of DOS, default: 2001.
        **kwargs: All kwargs supported by DictSet. Typically, user_incar_settings is a
            commonly used option.
    """
    mode: str = 'IPA'
    copy_wavecar: bool = True
    nbands_factor: float = 2
    reciprocal_density: float = 400
    nkred: tuple[int, int, int] | None = None
    nedos: int = 2001
    inherit_incar: bool = True
    force_gamma: bool = True
    CONFIG = MPRelaxSet.CONFIG
    nbands: int | None = None
    SUPPORTED_MODES = ('IPA', 'RPA')

    def __post_init__(self):
        """Validate settings"""
        super().__post_init__()
        self.mode = self.mode.upper()
        if self.mode not in MPAbsorptionSet.SUPPORTED_MODES:
            raise ValueError(f'{self.mode} not one of the support modes : {MPAbsorptionSet.SUPPORTED_MODES}')

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type.

        Generate gamma center k-points mesh grid for optical calculation. It is not
        mandatory for 'ALGO = Exact', but is requested by 'ALGO = CHI' calculation.
        """
        return {'reciprocal_density': self.reciprocal_density}

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = {'ALGO': 'Exact', 'EDIFF': 1e-08, 'IBRION': -1, 'ICHARG': 1, 'ISMEAR': 0, 'SIGMA': 0.01, 'LWAVE': True, 'LREAL': False, 'NELM': 100, 'NSW': 0, 'LOPTICS': True, 'CSHIFT': 0.1, 'NEDOS': self.nedos}
        if self.mode == 'RPA':
            updates.update({'ALGO': 'CHI', 'NELM': 1, 'NOMEGA': 1000, 'EDIFF': None, 'LOPTICS': None, 'LWAVE': None})
        if self.prev_vasprun is not None and self.mode == 'IPA':
            prev_nbands = int(self.prev_vasprun.parameters['NBANDS']) if self.nbands is None else self.nbands
            updates['NBANDS'] = int(np.ceil(prev_nbands * self.nbands_factor))
        if self.prev_vasprun is not None and self.mode == 'RPA':
            self.nkred = self.prev_vasprun.kpoints.kpts[0] if self.nkred is None else self.nkred
            updates.update({'NKREDX': self.nkred[0], 'NKREDY': self.nkred[1], 'NKREDZ': self.nkred[2]})
        return updates