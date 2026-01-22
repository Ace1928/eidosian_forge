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
def BE_vs_clean_SE(self, delu_dict, delu_default=0, plot_eads=False, annotate_monolayer=True, JPERM2=False):
    """
        For each facet, plot the clean surface energy against the most
            stable binding energy.

        Args:
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            plot_eads (bool): Option to plot the adsorption energy (binding
                energy multiplied by number of adsorbates) instead.
            annotate_monolayer (bool): Whether or not to label each data point
                with its monolayer (adsorbate density per unit primiitve area)
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False)

        Returns:
            Plot: Plot of clean surface energy vs binding energy for
                all facets.
        """
    ax = pretty_plot(width=8, height=7)
    for hkl in self.all_slab_entries:
        for clean_entry in self.all_slab_entries[hkl]:
            all_delu_dict = self.set_all_variables(delu_dict, delu_default)
            if self.all_slab_entries[hkl][clean_entry]:
                clean_se = self.as_coeffs_dict[clean_entry]
                se = sub_chempots(clean_se, all_delu_dict)
                for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                    ml = ads_entry.get_monolayer
                    be = ads_entry.gibbs_binding_energy(eads=plot_eads)
                    ax.scatter(se, be)
                    if annotate_monolayer:
                        ax.annotate(f'{ml:.2f}', xy=[se, be], xytext=[se, be])
    ax.set_xlabel('Surface energy ($J/m^2$)' if JPERM2 else 'Surface energy ($eV/\\AA^2$)')
    ax.set_ylabel('Adsorption Energy (eV)' if plot_eads else 'Binding Energy (eV)')
    plt.tight_layout()
    ax.set_xticks(rotation=60)
    return ax