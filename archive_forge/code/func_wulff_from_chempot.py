from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def wulff_from_chempot(self, delu_dict=None, delu_default=0, symprec=1e-05, no_clean=False, no_doped=False):
    """
        Method to get the Wulff shape at a specific chemical potential.

        Args:
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            symprec (float): See WulffShape.
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.

        Returns:
            WulffShape: The WulffShape at u_ref and u_ads.
        """
    lattice = SpacegroupAnalyzer(self.ucell_entry.structure).get_conventional_standard_structure().lattice
    miller_list = list(self.all_slab_entries)
    e_surf_list = []
    for hkl in miller_list:
        gamma = self.get_stable_entry_at_u(hkl, delu_dict=delu_dict, delu_default=delu_default, no_clean=no_clean, no_doped=no_doped)[1]
        e_surf_list.append(gamma)
    return WulffShape(lattice, miller_list, e_surf_list, symprec=symprec)