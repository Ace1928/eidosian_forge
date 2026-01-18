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

        Returns the plot of the formation energy of a particles
            of different polymorphs against its effect radius.

        Args:
            max_r (float): The maximum radius of the particle to plot up to.
            increments (int): Number of plot points
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            plt (pyplot): Plot
            labels (list): List of labels for each plot, corresponds to the
                list of se_analyzers
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.

        Returns:
            plt.Axes: matplotlib Axes object
        