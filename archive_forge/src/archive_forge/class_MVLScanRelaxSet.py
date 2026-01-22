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
class MVLScanRelaxSet(DictSet):
    """
    Class for writing a relax input set using Strongly Constrained and
    Appropriately Normed (SCAN) semilocal density functional.

    Notes:
        1. This functional is only available from VASP.5.4.3 upwards.

        2. Meta-GGA calculations require POTCAR files that include
        information on the kinetic energy density of the core-electrons,
        i.e. "PBE_52" or "PBE_54". Make sure the POTCAR including the
        following lines (see VASP wiki for more details):

            $ grep kinetic POTCAR
            kinetic energy-density
            mkinetic energy-density pseudized
            kinetic energy density (partial)

    Args:
        structure (Structure): input structure.
        vdw (str): set "rVV10" to enable SCAN+rVV10, which is a versatile
            van der Waals density functional by combing the SCAN functional
            with the rVV10 non-local correlation functional.
        **kwargs: Other kwargs supported by DictSet.
    """
    user_potcar_functional: UserPotcarFunctional = 'PBE_52'
    _valid_potcars = ('PBE_52', 'PBE_54')
    CONFIG = MPRelaxSet.CONFIG

    def __post_init__(self):
        super().__post_init__()
        if self.user_potcar_functional not in ('PBE_52', 'PBE_54'):
            raise ValueError('SCAN calculations required PBE_52 or PBE_54!')

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = {'ADDGRID': True, 'EDIFF': 1e-05, 'EDIFFG': -0.05, 'LASPH': True, 'LDAU': False, 'METAGGA': 'SCAN', 'NELM': 200}
        if self.vdw and self.vdw.lower() == 'rvv10':
            updates['BPARAM'] = 15.7
        return updates