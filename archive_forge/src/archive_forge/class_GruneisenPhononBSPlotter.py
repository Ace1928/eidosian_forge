from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
class GruneisenPhononBSPlotter(PhononBSPlotter):
    """Class to plot or get data to facilitate the plot of band structure objects."""

    def __init__(self, bs: GruneisenPhononBandStructureSymmLine) -> None:
        """
        Args:
            bs: A GruneisenPhononBandStructureSymmLine object.
        """
        if not isinstance(bs, GruneisenPhononBandStructureSymmLine):
            raise ValueError("GruneisenPhononBSPlotter only works with GruneisenPhononBandStructureSymmLine objects. A GruneisenPhononBandStructure object (on a uniform grid for instance and not along symmetry lines won't work)")
        super().__init__(bs)

    def bs_plot_data(self) -> dict[str, Any]:
        """Get the data nicely formatted for a plot.

        Returns:
            A dict of the following format:
            ticks: A dict with the 'distances' at which there is a qpoint (the
            x axis) and the labels (None if no label)
            frequencies: A list (one element for each branch) of frequencies for
            each qpoint: [branch][qpoint][mode]. The data is
            stored by branch to facilitate the plotting
            gruneisen: GruneisenPhononBandStructureSymmLine
            lattice: The reciprocal lattice.
        """
        distance: list = []
        frequency: list[list[list[float]]] = []
        gruneisen: list = []
        ticks = self.get_ticks()
        for branch in self._bs.branches:
            frequency.append([])
            gruneisen.append([])
            distance.append([self._bs.distance[j] for j in range(branch['start_index'], branch['end_index'] + 1)])
            for idx in range(self.n_bands):
                frequency[-1].append([self._bs.bands[idx][j] for j in range(branch['start_index'], branch['end_index'] + 1)])
                gruneisen[-1].append([self._bs.gruneisen[idx][j] for j in range(branch['start_index'], branch['end_index'] + 1)])
        return {'ticks': ticks, 'distances': distance, 'frequency': frequency, 'gruneisen': gruneisen, 'lattice': self._bs.lattice_rec.as_dict()}

    def get_plot_gs(self, ylim: float | None=None, **kwargs) -> Axes:
        """Get a matplotlib object for the Gruneisen bandstructure plot.

        Args:
            ylim: Specify the y-axis (gruneisen) limits; by default None let
                the code choose.
            **kwargs: additional keywords passed to ax.plot().
        """
        ax = pretty_plot(12, 8)
        kwargs.setdefault('linewidth', 2)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('markersize', 2)
        data = self.bs_plot_data()
        for dist_idx in range(len(data['distances'])):
            for band_idx in range(self.n_bands):
                ys = [data['gruneisen'][dist_idx][band_idx][idx] for idx in range(len(data['distances'][dist_idx]))]
                ax.plot(data['distances'][dist_idx], ys, 'b-', **kwargs)
        self._make_ticks(ax)
        ax.axhline(0, linewidth=1, color='black')
        ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=30)
        ax.set_ylabel('$\\mathrm{Grüneisen\\ Parameter}$', fontsize=30)
        x_max = data['distances'][-1][-1]
        ax.set_xlim(0, x_max)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()
        return ax

    def show_gs(self, ylim: float | None=None) -> None:
        """Show the plot using matplotlib.

        Args:
            ylim: Specifies the y-axis limits.
        """
        self.get_plot_gs(ylim)
        plt.show()

    def save_plot_gs(self, filename: str | PathLike, img_format: str='eps', ylim: float | None=None) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            ylim: Specifies the y-axis limits.
        """
        self.get_plot_gs(ylim=ylim)
        plt.savefig(filename, format=img_format)
        plt.close()

    def plot_compare_gs(self, other_plotter: GruneisenPhononBSPlotter) -> Axes:
        """Plot two band structure for comparison. One is in red the other in blue.
        The two band structures need to be defined on the same symmetry lines!
        and the distance between symmetry lines is
        the one of the band structure used to build the PhononBSPlotter.

        Args:
            other_plotter (GruneisenPhononBSPlotter): another phonon DOS plotter defined along
                the same symmetry lines.

        Raises:
            ValueError: if the two plotters are incompatible (due to different data lengths)

        Returns:
            a matplotlib object with both band structures
        """
        data_orig = self.bs_plot_data()
        data = other_plotter.bs_plot_data()
        len_orig = len(data_orig['distances'])
        len_other = len(data['distances'])
        if len_orig != len_other:
            raise ValueError(f'The two plotters are incompatible, plotting data have different lengths ({len_orig} vs {len_other}).')
        ax = self.get_plot()
        band_linewidth = 1
        for band_idx in range(other_plotter.n_bands):
            for dist_idx in range(len(data_orig['distances'])):
                ax.plot(data_orig['distances'][dist_idx], [data['gruneisen'][dist_idx][band_idx][j] for j in range(len(data_orig['distances'][dist_idx]))], 'r-', linewidth=band_linewidth)
        return ax