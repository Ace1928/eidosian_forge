from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
class PourbaixPlotter:
    """A plotter class for phase diagrams."""

    def __init__(self, pourbaix_diagram):
        """
        Args:
            pourbaix_diagram (PourbaixDiagram): A PourbaixDiagram object.
        """
        self._pbx = pourbaix_diagram

    def show(self, *args, **kwargs):
        """
        Shows the Pourbaix plot.

        Args:
            *args: args to get_pourbaix_plot
            **kwargs: kwargs to get_pourbaix_plot
        """
        plt = self.get_pourbaix_plot(*args, **kwargs)
        plt.show()

    @no_type_check
    def get_pourbaix_plot(self, limits: tuple[float, float] | None=None, title: str='', label_domains: bool=True, label_fontsize: int=20, show_water_lines: bool=True, show_neutral_axes: bool=True, ax: plt.Axes=None) -> plt.Axes:
        """
        Plot Pourbaix diagram.

        Args:
            limits: 2D list containing limits of the Pourbaix diagram
                of the form [[xlo, xhi], [ylo, yhi]]
            title (str): Title to display on plot
            label_domains (bool): whether to label Pourbaix domains
            label_fontsize: font size for domain labels
            show_water_lines: whether to show dashed lines indicating the region
                of water stability.
            show_neutral_axes; whether to show dashed horizontal and vertical lines
                at 0 V and pH 7, respectively.
            ax (Axes): Matplotlib Axes instance for plotting

        Returns:
            Axes: matplotlib Axes object with Pourbaix diagram
        """
        if limits is None:
            limits = [[-2, 16], [-3, 3]]
        ax = ax or pretty_plot(16)
        xlim, ylim = limits
        lw = 3
        if show_water_lines:
            h_line = np.transpose([[xlim[0], -xlim[0] * PREFAC], [xlim[1], -xlim[1] * PREFAC]])
            o_line = np.transpose([[xlim[0], -xlim[0] * PREFAC + 1.23], [xlim[1], -xlim[1] * PREFAC + 1.23]])
            ax.plot(h_line[0], h_line[1], 'r--', linewidth=lw)
            ax.plot(o_line[0], o_line[1], 'r--', linewidth=lw)
        if show_neutral_axes:
            neutral_line = np.transpose([[7, ylim[0]], [7, ylim[1]]])
            V0_line = np.transpose([[xlim[0], 0], [xlim[1], 0]])
            ax.plot(neutral_line[0], neutral_line[1], 'k-.', linewidth=lw)
            ax.plot(V0_line[0], V0_line[1], 'k-.', linewidth=lw)
        for entry, vertices in self._pbx._stable_domain_vertices.items():
            center = np.average(vertices, axis=0)
            x, y = np.transpose(np.vstack([vertices, vertices[0]]))
            ax.plot(x, y, 'k-', linewidth=lw)
            if label_domains:
                ax.annotate(generate_entry_label(entry), center, ha='center', va='center', fontsize=label_fontsize, color='b').draggable()
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set(xlabel='pH', ylabel='E (V)', xlim=xlim, ylim=ylim)
        return ax

    @no_type_check
    def plot_entry_stability(self, entry: Any, pH_range: tuple[float, float]=(-2, 16), pH_resolution: int=100, V_range: tuple[float, float]=(-3, 3), V_resolution: int=100, e_hull_max: float=1, cmap: str='RdYlBu_r', ax: plt.Axes | None=None, **kwargs: Any) -> plt.Axes:
        """
        Plots the stability of an entry in the Pourbaix diagram.

        Args:
            entry (Any): The entry to plot stability for.
            pH_range (tuple[float, float], optional): pH range for the plot. Defaults to (-2, 16).
            pH_resolution (int, optional): pH resolution. Defaults to 100.
            V_range (tuple[float, float], optional): Voltage range for the plot. Defaults to (-3, 3).
            V_resolution (int, optional): Voltage resolution. Defaults to 100.
            e_hull_max (float, optional): Maximum energy above the hull. Defaults to 1.
            cmap (str, optional): Colormap for the plot. Defaults to "RdYlBu_r".
            ax (Axes, optional): Existing matplotlib Axes object for plotting. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to `get_pourbaix_plot`.

        Returns:
            plt.Axes: Matplotlib Axes object with the plotted stability.
        """
        ax = self.get_pourbaix_plot(ax=ax, **kwargs)
        pH, V = np.mgrid[pH_range[0]:pH_range[1]:pH_resolution * 1j, V_range[0]:V_range[1]:V_resolution * 1j]
        stability = self._pbx.get_decomposition_energy(entry, pH, V)
        cax = ax.pcolor(pH, V, stability, cmap=cmap, vmin=0, vmax=e_hull_max)
        cbar = ax.figure.colorbar(cax)
        cbar.set_label(f'Stability of {generate_entry_label(entry)} (eV/atom)')
        return ax

    def domain_vertices(self, entry):
        """
        Returns the vertices of the Pourbaix domain.

        Args:
            entry: Entry for which domain vertices are desired

        Returns:
            list of vertices
        """
        return self._pbx._stable_domain_vertices[entry]