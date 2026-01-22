from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def _prep_calculator(self, calculator: Literal['m3gnet', 'gfn2-xtb'] | Calculator, **params) -> Calculator:
    """Convert string name of special ASE calculators into ASE calculator objects.

        Args:
            calculator: An ASE Calculator or a string from the following options: "m3gnet",
                "gfn2-xtb".
            **params: Parameters for the calculator.

        Returns:
            Calculator: ASE calculator object.
        """
    if inspect.isclass(calculator):
        return calculator(**params)
    if not isinstance(calculator, str):
        return calculator
    if calculator.lower() == 'chgnet':
        try:
            from chgnet.model import CHGNetCalculator
        except ImportError:
            raise ImportError('chgnet not installed. Try `pip install chgnet`.')
        return CHGNetCalculator()
    if calculator.lower() == 'm3gnet':
        try:
            import matgl
            from matgl.ext.ase import M3GNetCalculator
        except ImportError:
            raise ImportError('matgl not installed. Try `pip install matgl`.')
        potential = matgl.load_model('M3GNet-MP-2021.2.8-PES')
        return M3GNetCalculator(potential=potential, **params)
    if calculator.lower() == 'gfn2-xtb':
        try:
            from tblite.ase import TBLite
        except ImportError:
            raise ImportError('Must install tblite[ase]. Try `pip install tblite[ase]` (Linux)or `conda install -c conda-forge tblite-python` on (Mac/Windows).')
        return TBLite(method='GFN2-xTB', **params)
    raise ValueError(f'Unknown calculator={calculator!r}.')