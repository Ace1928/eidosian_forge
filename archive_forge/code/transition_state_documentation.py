from __future__ import annotations
import os
from glob import glob
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable, jsanitize
from scipy.interpolate import CubicSpline
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar
from pymatgen.util.plotting import pretty_plot

        Dict representation of NEBAnalysis.

        Returns:
            JSON-serializable dict representation.
        