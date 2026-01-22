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
def area_frac_vs_chempot_plot(self, ref_delu: Symbol, chempot_range: list[float], delu_dict: dict[Symbol, float] | None=None, delu_default: float=0, increments: int=10, no_clean: bool=False, no_doped: bool=False) -> plt.Axes:
    """
        1D plot. Plots the change in the area contribution
        of each facet as a function of chemical potential.

        Args:
            ref_delu (Symbol): The free variable chempot with the format:
                Symbol("delu_el") where el is the name of the element.
            chempot_range (list[float]): Min/max range of chemical potential to plot along.
            delu_dict (dict[Symbol, float]): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials.
            increments (int): Number of data points between min/max or point
                of intersection. Defaults to 10 points.
            no_clean (bool): Some parameter, description missing.
            no_doped (bool): Some parameter, description missing.

        Returns:
            plt.Axes: Plot of area frac on the Wulff shape for each facet vs chemical potential.
        """
    delu_dict = delu_dict or {}
    chempot_range = sorted(chempot_range)
    all_chempots = np.linspace(min(chempot_range), max(chempot_range), increments)
    hkl_area_dict: dict[tuple[int, int, int], list[float]] = {}
    for hkl in self.all_slab_entries:
        hkl_area_dict[hkl] = []
    for u in all_chempots:
        delu_dict[ref_delu] = u
        wulff_shape = self.wulff_from_chempot(delu_dict=delu_dict, no_clean=no_clean, no_doped=no_doped, delu_default=delu_default)
        for hkl in wulff_shape.area_fraction_dict:
            hkl_area_dict[hkl].append(wulff_shape.area_fraction_dict[hkl])
    ax = pretty_plot(width=8, height=7)
    for hkl in self.all_slab_entries:
        clean_entry = next(iter(self.all_slab_entries[hkl]))
        if all((a == 0 for a in hkl_area_dict[hkl])):
            continue
        plt.plot(all_chempots, hkl_area_dict[hkl], '--', color=self.color_dict[clean_entry], label=str(hkl))
    ax.set(ylabel='Fractional area $A^{Wulff}_{hkl}/A^{Wulff}$')
    self.chempot_plot_addons(ax, chempot_range, str(ref_delu).split('_')[1], rect=[-0.0, 0, 0.95, 1], pad=5, ylim=[0, 1])
    return ax