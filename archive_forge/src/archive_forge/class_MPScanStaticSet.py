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
class MPScanStaticSet(MPScanRelaxSet):
    """
    Creates input files for a static calculation using the accurate and numerically
    efficient r2SCAN variant of the Strongly Constrained and Appropriately Normed
    (SCAN) metaGGA functional.

    Args:
        structure (Structure): Structure from previous run.
        bandgap (float): Bandgap of the structure in eV. The bandgap is used to
            compute the appropriate k-point density and determine the smearing settings.
        lepsilon (bool): Whether to add static dielectric calculation
        lcalcpol (bool): Whether to turn on evaluation of the Berry phase approximations
            for electronic polarization.
        **kwargs: kwargs supported by MPScanRelaxSet.
    """
    lepsilon: bool = False
    lcalcpol: bool = False
    inherit_incar: bool = True

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates: dict[str, Any] = {'LREAL': False, 'NSW': 0, 'LORBIT': 11, 'LVHAR': True, 'ISMEAR': -5, 'KSPACING': 'auto'}
        if self.lepsilon:
            updates.update({'IBRION': 8, 'LEPSILON': True, 'LPEAD': True, 'NSW': 1, 'NPAR': None})
        if self.lcalcpol:
            updates['LCALCPOL'] = True
        return updates