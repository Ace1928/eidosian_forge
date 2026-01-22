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
class MPMDSet(DictSet):
    """
    This a modified version of the old MITMDSet pre 2018/03/12.

    This set serves as the basis for the amorphous skyline paper.

    (1) Aykol, M.; Dwaraknath, S. S.; Sun, W.; Persson, K. A. Thermodynamic
        Limit for Synthesis of Metastable Inorganic Materials. Sci. Adv. 2018,
        4 (4).

    Class for writing a vasp md run. This DOES NOT do multiple stage runs.
    Precision remains normal, to increase accuracy of stress tensor.

    Args:
        structure (Structure): Input structure.
        start_temp (int): Starting temperature.
        end_temp (int): Final temperature.
        nsteps (int): Number of time steps for simulations. NSW parameter.
        time_step (float): The time step for the simulation. The POTIM
            parameter. Defaults to None, which will set it automatically
            to 2.0 fs for non-hydrogen containing structures and 0.5 fs
            for hydrogen containing structures.
        spin_polarized (bool): Whether to do spin polarized calculations.
            The ISPIN parameter. Defaults to False.
        **kwargs: Other kwargs supported by DictSet.
    """
    start_temp: float = 0.0
    end_temp: float = 300.0
    nsteps: int = 1000
    time_step: float | None = None
    spin_polarized: bool = False
    CONFIG = MPRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = {'TEBEG': self.start_temp, 'TEEND': self.end_temp, 'NSW': self.nsteps, 'EDIFF_PER_ATOM': 1e-05, 'LSCALU': False, 'LCHARG': False, 'LPLANE': False, 'LWAVE': True, 'ISMEAR': 0, 'NELMIN': 4, 'LREAL': True, 'BMIX': 1, 'MAXMIX': 20, 'NELM': 500, 'NSIM': 4, 'ISYM': 0, 'ISIF': 0, 'IBRION': 0, 'NBLOCK': 1, 'KBLOCK': 100, 'SMASS': 0, 'PREC': 'Normal', 'ISPIN': 2 if self.spin_polarized else 1, 'LDAU': False, 'ADDGRID': True, 'ENCUT': None}
        if not self.spin_polarized:
            updates['MAGMOM'] = None
        if self.time_step is None:
            if Element('H') in self.structure.species:
                updates.update({'POTIM': 0.5, 'NSW': self.nsteps * 4})
            else:
                updates['POTIM'] = 2.0
        else:
            updates['POTIM'] = self.time_step
        return updates

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type."""
        return Kpoints.gamma_automatic()