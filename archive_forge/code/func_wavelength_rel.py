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
def wavelength_rel(self) -> float:
    """
        Calculates the wavelength of the electron beam with relativistic kinematic effects taken
            into account.

        Returns:
            float: Relativistic Wavelength (in angstroms)
        """
    sqr = 2 * sc.m_e * sc.e * 1000 * self.voltage * (1 + sc.e * 1000 * self.voltage / (2 * sc.m_e * sc.c ** 2))
    return sc.h / np.sqrt(sqr) * 10 ** 10