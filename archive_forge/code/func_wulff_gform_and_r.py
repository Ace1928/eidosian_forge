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
def wulff_gform_and_r(self, wulff_shape, bulk_entry, r, from_sphere_area=False, r_units='nanometers', e_units='keV', normalize=False, scale_per_atom=False):
    """
        Calculates the formation energy of the particle with arbitrary radius r.

        Args:
            wulff_shape (WulffShape): Initial unscaled WulffShape
            bulk_entry (ComputedStructureEntry): Entry of the corresponding bulk.
            r (float (Ang)): Arbitrary effective radius of the WulffShape
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.
            r_units (str): Can be nanometers or Angstrom
            e_units (str): Can be keV or eV
            normalize (bool): Whether or not to normalize energy by volume
            scale_per_atom (True): Whether or not to normalize by number of
                atoms in the particle

        Returns:
            particle formation energy (float in keV), effective radius
        """
    miller_se_dict = wulff_shape.miller_energy_dict
    new_wulff = self.scaled_wulff(wulff_shape, r)
    new_wulff_area = new_wulff.miller_area_dict
    if not from_sphere_area:
        w_vol = new_wulff.volume
        tot_wulff_se = 0
        for hkl, v in new_wulff_area.items():
            tot_wulff_se += miller_se_dict[hkl] * v
        Ebulk = self.bulk_gform(bulk_entry) * w_vol
        new_r = new_wulff.effective_radius
    else:
        w_vol = 4 / 3 * np.pi * r ** 3
        sphere_sa = 4 * np.pi * r ** 2
        tot_wulff_se = wulff_shape.weighted_surface_energy * sphere_sa
        Ebulk = self.bulk_gform(bulk_entry) * w_vol
        new_r = r
    new_r = new_r / 10 if r_units == 'nanometers' else new_r
    e = Ebulk + tot_wulff_se
    e = e / 1000 if e_units == 'keV' else e
    e = e / (4 / 3 * np.pi * new_r ** 3) if normalize else e
    bulk_struct = bulk_entry.structure
    density = len(bulk_struct) / bulk_struct.volume
    e = e / (density * w_vol) if scale_per_atom else e
    return (e, new_r)