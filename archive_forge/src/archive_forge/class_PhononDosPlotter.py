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
class PhononDosPlotter:
    """Class for plotting phonon DOSs. The interface is extremely flexible given there are many
    different ways in which people want to view DOS.
    Typical usage is:
        # Initializes plotter with some optional args. Defaults are usually fine
        plotter = PhononDosPlotter().

        # Add DOS with a label
        plotter.add_dos("Total DOS", dos)

        # Alternatively, you can add a dict of DOSes. This is the typical form
        # returned by CompletePhononDos.get_element_dos().
    """

    def __init__(self, stack: bool=False, sigma: float | None=None) -> None:
        """
        Args:
            stack: Whether to plot the DOS as a stacked area graph
            sigma: A float specifying a standard deviation for Gaussian smearing
                the DOS for nicer looking plots. Defaults to None for no smearing.
        """
        if not isinstance(stack, bool):
            raise ValueError('The first argument stack expects a boolean. If you are trying to add a DOS, use the add_dos() method.')
        self.stack = stack
        self.sigma = sigma
        self._doses: dict[str, dict[str, np.ndarray]] = {}

    def add_dos(self, label: str, dos: PhononDos, **kwargs: Any) -> None:
        """Adds a dos for plotting.

        Args:
            label (str): label for the DOS. Must be unique.
            dos (PhononDos): DOS object
            **kwargs: kwargs supported by matplotlib.pyplot.plot
        """
        densities = dos.get_smeared_densities(self.sigma) if self.sigma else dos.densities
        self._doses[label] = {'frequencies': dos.frequencies, 'densities': densities, **kwargs}

    def add_dos_dict(self, dos_dict: dict, key_sort_func=None) -> None:
        """Add a dictionary of doses, with an optional sorting function for the
        keys.

        Args:
            dos_dict: dict of {label: Dos}
            key_sort_func: function used to sort the dos_dict keys.
        """
        keys = sorted(dos_dict, key=key_sort_func) if key_sort_func else list(dos_dict)
        for label in keys:
            self.add_dos(label, dos_dict[label])

    def get_dos_dict(self) -> dict:
        """Returns the added doses as a json-serializable dict. Note that if you
        have specified smearing for the DOS plot, the densities returned will
        be the smeared densities, not the original densities.

        Returns:
            dict: DOS data. Generally of the form {label: {'frequencies':.., 'densities': ...}}
        """
        return jsanitize(self._doses)

    def get_plot(self, xlim: float | None=None, ylim: float | None=None, invert_axes: bool=False, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', legend: dict | None=None, ax: Axes | None=None) -> Axes:
        """Get a matplotlib plot showing the DOS.

        Args:
            xlim: Specifies the x-axis limits. Set to None for automatic determination.
            ylim: Specifies the y-axis limits.
            invert_axes (bool): Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units (thz | ev | mev | ha | cm-1 | cm^-1): units for the frequencies. Defaults to "thz".
            legend: dict with legend options. For example, {"loc": "upper right"}
                will place the legend in the upper right corner. Defaults to {"fontsize": 30}.
            ax (Axes): An existing axes object onto which the plot will be added.
                If None, a new figure will be created.
        """
        legend = legend or {}
        legend.setdefault('fontsize', 30)
        unit = freq_units(units)
        n_colors = max(3, len(self._doses))
        n_colors = min(9, n_colors)
        ys = None
        all_densities = []
        all_frequencies = []
        ax = pretty_plot(*((8, 12) if invert_axes else (12, 8)), ax=ax)
        for dos in self._doses.values():
            frequencies = dos['frequencies'] * unit.factor
            densities = dos['densities']
            if ys is None:
                ys = np.zeros(frequencies.shape)
            if self.stack:
                ys += densities
                new_dens = ys.copy()
            else:
                new_dens = densities
            all_frequencies.append(frequencies)
            all_densities.append(new_dens)
        keys = list(reversed(self._doses))
        all_densities.reverse()
        all_frequencies.reverse()
        all_pts = []
        colors = ('blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive')
        for idx, (key, frequencies, densities) in enumerate(zip(keys, all_frequencies, all_densities)):
            color = self._doses[key].get('color', colors[idx % n_colors])
            linewidth = self._doses[key].get('linewidth', 3)
            kwargs = {key: val for key, val in self._doses[key].items() if key not in ['frequencies', 'densities', 'color', 'linewidth']}
            all_pts.extend(list(zip(frequencies, densities)))
            if invert_axes:
                xs, ys = (densities, frequencies)
            else:
                xs, ys = (frequencies, densities)
            if self.stack:
                ax.fill(xs, ys, color=color, label=str(key), **kwargs)
            else:
                ax.plot(xs, ys, color=color, label=str(key), linewidth=linewidth, **kwargs)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        elif invert_axes:
            _ylim = ax.get_ylim()
            relevant_x = [p[1] for p in all_pts if _ylim[0] < p[0] < _ylim[1]] or ax.get_xlim()
            ax.set_xlim((min(relevant_x), max(relevant_x)))
        else:
            _xlim = ax.get_xlim()
            relevant_y = [p[1] for p in all_pts if _xlim[0] < p[0] < _xlim[1]] or ax.get_ylim()
            ax.set_ylim((min(relevant_y), max(relevant_y)))
        if invert_axes:
            ax.axhline(0, linewidth=2, color='black', linestyle='--')
            ax.set_xlabel('$\\mathrm{Density\\ of\\ states}$', fontsize=legend.get('fontsize', 30))
            ax.set_ylabel(f'$\\mathrm{{Frequencies\\ ({unit.label})}}$', fontsize=legend.get('fontsize', 30))
        else:
            ax.axvline(0, linewidth=2, color='black', linestyle='--')
            ax.set_xlabel(f'$\\mathrm{{Frequencies\\ ({unit.label})}}$', fontsize=legend.get('fontsize', 30))
            ax.set_ylabel('$\\mathrm{Density\\ of\\ states}$', fontsize=legend.get('fontsize', 30))
        if sum(map(len, keys)) > 0:
            ax.legend(**legend)
        return ax

    def save_plot(self, filename: str | PathLike, img_format: str='eps', xlim: float | None=None, ylim: float | None=None, invert_axes: bool=False, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Save matplotlib plot to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes: Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1
        """
        self.get_plot(xlim, ylim, invert_axes=invert_axes, units=units)
        plt.savefig(filename, format=img_format)
        plt.close()

    def show(self, xlim: float | None=None, ylim: None=None, invert_axes: bool=False, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Show the plot using matplotlib.

        Args:
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes: Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(xlim, ylim, invert_axes=invert_axes, units=units)
        plt.show()