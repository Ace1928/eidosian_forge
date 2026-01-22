from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
class CohpPlotter:
    """Class for plotting crystal orbital Hamilton populations (COHPs) or
    crystal orbital overlap populations (COOPs). It is modeled after the
    DosPlotter object.
    """

    def __init__(self, zero_at_efermi=True, are_coops=False, are_cobis=False) -> None:
        """
        Args:
            zero_at_efermi: Whether to shift all populations to have zero
                energy at the Fermi level. Defaults to True.
            are_coops: Switch to indicate that these are COOPs, not COHPs.
                Defaults to False for COHPs.
            are_cobis: Switch to indicate that these are COBIs or multi-center COBIs, not COHPs/COOPs.
                Defaults to False for COHPs.
        """
        self.zero_at_efermi = zero_at_efermi
        self.are_coops = are_coops
        self.are_cobis = are_cobis
        self._cohps: dict[str, dict[str, np.ndarray | dict[Spin, np.ndarray] | float]] = {}

    def add_cohp(self, label, cohp) -> None:
        """Adds a COHP for plotting.

        Args:
            label: Label for the COHP. Must be unique.

            cohp: COHP object.
        """
        energies = cohp.energies - cohp.efermi if self.zero_at_efermi else cohp.energies
        populations = cohp.get_cohp()
        int_populations = cohp.get_icohp()
        self._cohps[label] = {'energies': energies, 'COHP': populations, 'ICOHP': int_populations, 'efermi': cohp.efermi}

    def add_cohp_dict(self, cohp_dict, key_sort_func=None) -> None:
        """Adds a dictionary of COHPs with an optional sorting function
        for the keys.

        Args:
            cohp_dict: dict of the form {label: Cohp}

            key_sort_func: function used to sort the cohp_dict keys.
        """
        keys = sorted(cohp_dict, key=key_sort_func) if key_sort_func else list(cohp_dict)
        for label in keys:
            self.add_cohp(label, cohp_dict[label])

    def get_cohp_dict(self):
        """Returns the added COHPs as a json-serializable dict. Note that if you
        have specified smearing for the COHP plot, the populations returned
        will be the smeared and not the original populations.

        Returns:
            dict: Dict of COHP data of the form {label: {"efermi": efermi,
            "energies": ..., "COHP": {Spin.up: ...}, "ICOHP": ...}}.
        """
        return jsanitize(self._cohps)

    def get_plot(self, xlim=None, ylim=None, plot_negative=None, integrated=False, invert_axes=True):
        """Get a matplotlib plot showing the COHP.

        Args:
            xlim: Specifies the x-axis limits. Defaults to None for
                automatic determination.

            ylim: Specifies the y-axis limits. Defaults to None for
                automatic determination.

            plot_negative: It is common to plot -COHP(E) so that the
                sign means the same for COOPs and COHPs. Defaults to None
                for automatic determination: If are_coops is True, this
                will be set to False, else it will be set to True.

            integrated: Switch to plot ICOHPs. Defaults to False.

            invert_axes: Put the energies onto the y-axis, which is
                common in chemistry.

        Returns:
            A matplotlib object.
        """
        if self.are_coops:
            cohp_label = 'COOP'
        elif self.are_cobis:
            cohp_label = 'COBI'
        else:
            cohp_label = 'COHP'
        if plot_negative is None:
            plot_negative = not self.are_coops and (not self.are_cobis)
        if integrated:
            cohp_label = f'I{cohp_label} (eV)'
        if plot_negative:
            cohp_label = f'-{cohp_label}'
        energy_label = '$E - E_f$ (eV)' if self.zero_at_efermi else '$E$ (eV)'
        ncolors = max(3, len(self._cohps))
        ncolors = min(9, ncolors)
        colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax = pretty_plot(12, 8)
        allpts = []
        keys = list(self._cohps)
        for i, key in enumerate(keys):
            energies = self._cohps[key]['energies']
            populations = self._cohps[key]['COHP'] if not integrated else self._cohps[key]['ICOHP']
            for spin in [Spin.up, Spin.down]:
                if spin in populations:
                    if invert_axes:
                        x = -populations[spin] if plot_negative else populations[spin]
                        y = energies
                    else:
                        x = energies
                        y = -populations[spin] if plot_negative else populations[spin]
                    allpts.extend(list(zip(x, y)))
                    if spin == Spin.up:
                        ax.plot(x, y, color=colors[i % ncolors], linestyle='-', label=str(key), linewidth=3)
                    else:
                        ax.plot(x, y, color=colors[i % ncolors], linestyle='--', linewidth=3)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        elif not invert_axes:
            xlim = ax.get_xlim()
            relevant_y = [p[1] for p in allpts if xlim[0] < p[0] < xlim[1]]
            ax.set_ylim((min(relevant_y), max(relevant_y)))
        if not xlim and invert_axes:
            ylim = ax.get_ylim()
            relevant_y = [p[0] for p in allpts if ylim[0] < p[1] < ylim[1]]
            ax.set_xlim((min(relevant_y), max(relevant_y)))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if not invert_axes:
            ax.axhline(y=0, color='k', linewidth=2)
            if self.zero_at_efermi:
                ax.plot([0, 0], ylim, 'k--', linewidth=2)
            else:
                ax.plot([self._cohps[key]['efermi'], self._cohps[key]['efermi']], ylim, color=colors[i % ncolors], linestyle='--', linewidth=2)
        else:
            ax.axvline(x=0, color='k', linewidth=2)
            if self.zero_at_efermi:
                ax.plot(xlim, [0, 0], 'k--', linewidth=2)
            else:
                ax.plot(xlim, [self._cohps[key]['efermi'], self._cohps[key]['efermi']], color=colors[i % ncolors], linestyle='--', linewidth=2)
        if invert_axes:
            ax.set_xlabel(cohp_label)
            ax.set_ylabel(energy_label)
        else:
            ax.set_xlabel(energy_label)
            ax.set_ylabel(cohp_label)
        ax.legend()
        legend_text = ax.legend().get_texts()
        plt.setp(legend_text, fontsize=30)
        plt.tight_layout()
        return ax

    def save_plot(self, filename: str, xlim=None, ylim=None) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename (str): File name to write to. Must include extension to specify image format.
            xlim: Specifies the x-axis limits. Defaults to None for
                automatic determination.
            ylim: Specifies the y-axis limits. Defaults to None for
                automatic determination.
        """
        self.get_plot(xlim, ylim)
        plt.savefig(filename)

    def show(self, xlim=None, ylim=None) -> None:
        """Show the plot using matplotlib.

        Args:
            xlim: Specifies the x-axis limits. Defaults to None for
                automatic determination.
            ylim: Specifies the y-axis limits. Defaults to None for
                automatic determination.
        """
        self.get_plot(xlim, ylim)
        plt.show()