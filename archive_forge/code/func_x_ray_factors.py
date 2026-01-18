from __future__ import annotations
import json
import os
from collections import namedtuple
from fractions import Fraction
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants as sc
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup, unicodeify_spacegroup
def x_ray_factors(self, structure: Structure, bragg_angles: dict[tuple[int, int, int], float]) -> dict[str, dict[tuple[int, int, int], float]]:
    """
        Calculates x-ray factors, which are required to calculate atomic scattering factors. Method partially inspired
        by the equivalent process in the xrd module.

        Args:
            structure (Structure): The input structure.
            bragg_angles (dict): Dictionary of hkl plane to Bragg angle.

        Returns:
            dict of atomic symbol to another dict of hkl plane to x-ray factor (in angstroms).
        """
    x_ray_factors = {}
    s2 = self.get_s2(bragg_angles)
    atoms = structure.elements
    scattering_factors_for_atom = {}
    for atom in atoms:
        coeffs = np.array(ATOMIC_SCATTERING_PARAMS[atom.symbol])
        for plane in bragg_angles:
            scattering_factor_curr = atom.Z - 41.78214 * s2[plane] * np.sum(coeffs[:, 0] * np.exp(-coeffs[:, 1] * s2[plane]), axis=None)
            scattering_factors_for_atom[plane] = scattering_factor_curr
        x_ray_factors[atom.symbol] = scattering_factors_for_atom
        scattering_factors_for_atom = {}
    return x_ray_factors