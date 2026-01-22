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
@due.dcite(Doi('10.1149/2.0061602jes'), description='Elastic Properties of Alkali Superionic Conductor Electrolytes from First Principles Calculations')
class MVLElasticSet(DictSet):
    """
    MVL denotes VASP input sets that are implemented by the Materials Virtual
    Lab (http://materialsvirtuallab.org) for various research.

    This input set is used to calculate elastic constants in VASP. It is used
    in the following work::

        Z. Deng, Z. Wang, I.-H. Chu, J. Luo, S. P. Ong.
        “Elastic Properties of Alkali Superionic Conductor Electrolytes
        from First Principles Calculations”, J. Electrochem. Soc.
        2016, 163(2), A67-A74. doi: 10.1149/2.0061602jes

    To read the elastic constants, you may use the Outcar class which parses the
    elastic constants.

    Args:
        structure (pymatgen.Structure): Input structure.
        potim (float): POTIM parameter. The default of 0.015 is usually fine,
            but some structures may require a smaller step.
        **kwargs: Parameters supported by MPRelaxSet.
    """
    potim: float = 0.015
    CONFIG = MPRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        return {'IBRION': 6, 'NFREE': 2, 'POTIM': self.potim, 'NPAR': None}