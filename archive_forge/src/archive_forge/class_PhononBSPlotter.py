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
class PhononBSPlotter:
    """Class to plot or get data to facilitate the plot of band structure objects."""

    def __init__(self, bs: PhononBandStructureSymmLine, label: str | None=None) -> None:
        """
        Args:
            bs: A PhononBandStructureSymmLine object.
            label: A label for the plot. Defaults to None for no label. Esp. useful with
                the plot_compare method to distinguish the band structures.
        """
        if not isinstance(bs, PhononBandStructureSymmLine):
            raise ValueError("PhononBSPlotter only works with PhononBandStructureSymmLine objects. A PhononBandStructure object (on a uniform grid for instance and not along symmetry lines won't work)")
        self._bs = bs
        self._label = label

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return self._bs.nb_bands

    def _make_ticks(self, ax: Axes) -> Axes:
        """Utility private method to add ticks to a band structure."""
        ticks = self.get_ticks()
        ticks_labels = list(zip(*zip(ticks['distance'], ticks['label'])))
        if ticks_labels:
            ax.set_xticks(ticks_labels[0])
            ax.set_xticklabels(ticks_labels[1])
        for idx, label in enumerate(ticks['label']):
            if label is not None:
                ax.axvline(ticks['distance'][idx], color='black')
        return ax

    def bs_plot_data(self) -> dict[str, Any]:
        """Get the data nicely formatted for a plot.

        Returns:
            A dict of the following format:
            ticks: A dict with the 'distances' at which there is a qpoint (the
            x axis) and the labels (None if no label)
            frequencies: A list (one element for each branch) of frequencies for
            each qpoint: [branch][qpoint][mode]. The data is
            stored by branch to facilitate the plotting
            lattice: The reciprocal lattice.
        """
        distance = []
        frequency: list = []
        ticks = self.get_ticks()
        for branch in self._bs.branches:
            frequency.append([])
            distance.append([self._bs.distance[j] for j in range(branch['start_index'], branch['end_index'] + 1)])
            for idx in range(self.n_bands):
                frequency[-1].append([self._bs.bands[idx][j] for j in range(branch['start_index'], branch['end_index'] + 1)])
        return {'ticks': ticks, 'distances': distance, 'frequency': frequency, 'lattice': self._bs.lattice_rec.as_dict()}

    def get_plot(self, ylim: float | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', **kwargs) -> Axes:
        """Get a matplotlib object for the bandstructure plot.

        Args:
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            **kwargs: passed to ax.plot function.
        """
        u = freq_units(units)
        ax = pretty_plot(12, 8)
        data = self.bs_plot_data()
        kwargs.setdefault('color', 'blue')
        for dists, freqs in zip(data['distances'], data['frequency']):
            for idx in range(self.n_bands):
                ys = [freqs[idx][j] * u.factor for j in range(len(dists))]
                ax.plot(dists, ys, **kwargs)
        self._make_ticks(ax)
        ax.axhline(0, linewidth=1, color='black')
        ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=30)
        ylabel = f'$\\mathrm{{Frequencies\\ ({u.label})}}$'
        ax.set_ylabel(ylabel, fontsize=30)
        x_max = data['distances'][-1][-1]
        ax.set_xlim(0, x_max)
        if ylim is not None:
            ax.set_ylim(ylim)
        return ax

    def _get_weight(self, vec: np.ndarray, indices: list[list[int]]) -> np.ndarray:
        """Compute the weight for each combination of sites according to the
        eigenvector.
        """
        num_atom = int(self.n_bands / 3)
        new_vec = np.zeros(num_atom)
        for idx in range(num_atom):
            new_vec[idx] = np.linalg.norm(vec[idx * 3:idx * 3 + 3])
        gw = []
        norm_f = 0
        for comb in indices:
            projector = np.zeros(len(new_vec))
            for idx in range(len(projector)):
                if idx in comb:
                    projector[idx] = 1
            group_weight = np.dot(projector, new_vec)
            gw.append(group_weight)
            norm_f += group_weight
        return np.array(gw, dtype=float) / norm_f

    @staticmethod
    def _make_color(colors: Sequence[int]) -> Sequence[int]:
        """Convert the eigen-displacements to rgb colors."""
        if len(colors) == 2:
            return [colors[0], 0, colors[1]]
        if len(colors) == 3:
            return colors
        if len(colors) == 4:
            red = (1 - colors[0]) * (1 - colors[3])
            green = (1 - colors[1]) * (1 - colors[3])
            blue = (1 - colors[2]) * (1 - colors[3])
            return [red, green, blue]
        raise ValueError(f'Expected 2, 3 or 4 colors, got {len(colors)}')

    def get_proj_plot(self, site_comb: str | list[list[int]]='element', ylim: tuple[None | float, None | float] | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', rgb_labels: tuple[None | str] | None=None) -> Axes:
        """Get a matplotlib object for the bandstructure plot projected along atomic
        sites.

        Args:
            site_comb: a list of list, for example, [[0],[1],[2,3,4]];
                the numbers in each sublist represents the indices of atoms;
                the atoms in a same sublist will be plotted in a same color;
                if not specified, unique elements are automatically grouped.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: a list of rgb colors for the labels; if not specified,
                the colors will be automatically generated.
        """
        assert self._bs.structure is not None, 'Structure is required for get_proj_plot'
        elements = [elem.symbol for elem in self._bs.structure.elements]
        if site_comb == 'element':
            assert 2 <= len(elements) <= 4, 'the compound must have 2, 3 or 4 unique elements'
            indices: list[list[int]] = [[] for _ in range(len(elements))]
            for idx, elem in enumerate(self._bs.structure.species):
                for j, unique_species in enumerate(self._bs.structure.elements):
                    if elem == unique_species:
                        indices[j].append(idx)
        else:
            assert isinstance(site_comb, list)
            assert 2 <= len(site_comb) <= 4, 'the length of site_comb must be 2, 3 or 4'
            all_sites = self._bs.structure.sites
            all_indices = {*range(len(all_sites))}
            for comb in site_comb:
                for idx in comb:
                    assert 0 <= idx < len(all_sites), 'one or more indices in site_comb does not exist'
                    all_indices.remove(idx)
            if len(all_indices) != 0:
                raise Exception(f'not all {len(all_sites)} indices are included in site_comb')
            indices = site_comb
        assert rgb_labels is None or len(rgb_labels) == len(indices), 'wrong number of rgb_labels'
        u = freq_units(units)
        _fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        self._make_ticks(ax)
        data = self.bs_plot_data()
        k_dist = np.array(data['distances']).flatten()
        for d in range(1, len(k_dist)):
            colors = []
            for idx in range(self.n_bands):
                eigenvec_1 = self._bs.eigendisplacements[idx][d - 1].flatten()
                eigenvec_2 = self._bs.eigendisplacements[idx][d].flatten()
                colors1 = self._get_weight(eigenvec_1, indices)
                colors2 = self._get_weight(eigenvec_2, indices)
                colors.append(self._make_color((colors1 + colors2) / 2))
            seg = np.zeros((self.n_bands, 2, 2))
            seg[:, :, 1] = self._bs.bands[:, d - 1:d + 1] * u.factor
            seg[:, 0, 0] = k_dist[d - 1]
            seg[:, 1, 0] = k_dist[d]
            ls = LineCollection(seg, colors=colors, linestyles='-', linewidths=2.5)
            ax.add_collection(ls)
        if ylim is None:
            y_max: float = max((max(b) for b in self._bs.bands)) * u.factor
            y_min: float = min((min(b) for b in self._bs.bands)) * u.factor
            y_margin = (y_max - y_min) * 0.05
            ylim = (y_min - y_margin, y_max + y_margin)
        ax.set_ylim(ylim)
        xlim = [min(k_dist), max(k_dist)]
        ax.set_xlim(xlim)
        ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=28)
        ylabel = f'$\\mathrm{{Frequencies\\ ({u.label})}}$'
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(labelsize=28)
        labels: list[str]
        if rgb_labels is not None:
            labels = rgb_labels
        elif site_comb == 'element':
            labels = [elem.symbol for elem in self._bs.structure.elements]
        else:
            labels = [f'{idx}' for idx in range(len(site_comb))]
        if len(indices) == 2:
            BSDOSPlotter._rb_line(ax, labels[0], labels[1], 'best')
        elif len(indices) == 3:
            BSDOSPlotter._rgb_triangle(ax, labels[0], labels[1], labels[2], 'best')
        else:
            pass
        return ax

    def show(self, ylim: float | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Show the plot using matplotlib.

        Args:
            ylim (float): Specifies the y-axis limits.
            units ("thz" | "ev" | "mev" | "ha" | "cm-1" | "cm^-1"): units for the frequencies.
        """
        self.get_plot(ylim, units=units)
        plt.show()

    def save_plot(self, filename: str | PathLike, ylim: float | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Save matplotlib plot to a file.

        Args:
            filename (str | Path): Filename to write to.
            ylim (float): Specifies the y-axis limits.
            units ("thz" | "ev" | "mev" | "ha" | "cm-1" | "cm^-1"): units for the frequencies.
        """
        self.get_plot(ylim=ylim, units=units)
        plt.savefig(filename)
        plt.close()

    def show_proj(self, site_comb: str | list[list[int]]='element', ylim: tuple[None | float, None | float] | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', rgb_labels: tuple[str] | None=None) -> None:
        """Show the projected plot using matplotlib.

        Args:
            site_comb: A list of list of indices of sites to combine. For example,
                [[0, 1], [2, 3]] will combine the projections of sites 0 and 1,
                and sites 2 and 3. Defaults to "element", which will combine
                sites by element.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: A list of labels for the rgb triangle. Defaults to None,
                which will use the element symbols.
        """
        self.get_proj_plot(site_comb=site_comb, ylim=ylim, units=units, rgb_labels=rgb_labels)
        plt.show()

    def get_ticks(self) -> dict[str, list]:
        """Get all ticks and labels for a band structure plot.

        Returns:
            A dict with 'distance': a list of distance at which ticks should
            be set and 'label': a list of label for each of those ticks.
        """
        tick_distance = []
        tick_labels: list[str] = []
        prev_label = self._bs.qpoints[0].label
        prev_branch = self._bs.branches[0]['name']
        for idx, point in enumerate(self._bs.qpoints):
            if point.label is not None:
                tick_distance.append(self._bs.distance[idx])
                this_branch = None
                for b in self._bs.branches:
                    if b['start_index'] <= idx <= b['end_index']:
                        this_branch = b['name']
                        break
                if point.label != prev_label and prev_branch != this_branch:
                    label1 = point.label
                    if label1.startswith('\\') or label1.find('_') != -1:
                        label1 = f'${label1}$'
                    label0 = prev_label or ''
                    if label0.startswith('\\') or label0.find('_') != -1:
                        label0 = f'${label0}$'
                    tick_labels.pop()
                    tick_distance.pop()
                    tick_labels.append(f'{label0}|{label1}')
                elif point.label.startswith('\\') or point.label.find('_') != -1:
                    tick_labels.append(f'${point.label}$')
                else:
                    tick_labels.append(point.label)
                prev_label = point.label
                prev_branch = this_branch
        tick_labels = [label.replace('GAMMA', 'Γ').replace('DELTA', 'Δ').replace('SIGMA', 'Σ') for label in tick_labels]
        return {'distance': tick_distance, 'label': tick_labels}

    def plot_compare(self, other_plotter: PhononBSPlotter | dict[str, PhononBSPlotter], units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', self_label: str='self', colors: Sequence[str] | None=None, legend_kwargs: dict | None=None, on_incompatible: Literal['raise', 'warn', 'ignore']='raise', other_kwargs: dict | None=None, **kwargs) -> Axes:
        """Plot two band structure for comparison. self in blue, others in red, green, ...
        The band structures need to be defined on the same symmetry lines!
        The distance between symmetry lines is determined by the band structure used to
        initialize PhononBSPlotter (self).

        Args:
            other_plotter (PhononBSPlotter | dict[str, PhononBSPlotter]): Other PhononBSPlotter object(s) defined along
                the same symmetry lines
            units (str): units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to 'thz'.
            self_label (str): label for the self band structure. Defaults to to the label passed to PhononBSPlotter.init
                or, if None, 'self'.
            colors (list[str]): list of colors for the other band structures. Defaults to None for automatic colors.
            legend_kwargs: dict[str, Any]: kwargs passed to ax.legend().
            on_incompatible ('raise' | 'warn' | 'ignore'): What to do if the band structures
                are not compatible. Defaults to 'raise'.
            other_kwargs: dict[str, Any]: kwargs passed to other_plotter ax.plot().
            **kwargs: passed to ax.plot().

        Returns:
            a matplotlib object with both band structures
        """
        unit = freq_units(units)
        legend_kwargs = legend_kwargs or {}
        other_kwargs = other_kwargs or {}
        legend_kwargs.setdefault('fontsize', 20)
        _colors = ('blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive')
        if isinstance(other_plotter, PhononBSPlotter):
            other_plotter = {other_plotter._label or 'other': other_plotter}
        if colors:
            assert len(colors) == len(other_plotter) + 1, 'Wrong number of colors'
        self_data = self.bs_plot_data()
        line_width = kwargs.setdefault('linewidth', 1)
        ax = self.get_plot(units=units, color=colors[0] if colors else _colors[0], **kwargs)
        colors_other = []
        for idx, plotter in enumerate(other_plotter.values()):
            other_data = plotter.bs_plot_data()
            if np.asarray(self_data['distances']).shape != np.asarray(other_data['distances']).shape:
                if on_incompatible == 'raise':
                    raise ValueError('The two band structures are not compatible.')
                if on_incompatible == 'warn':
                    logger.warning('The two band structures are not compatible.')
                return None
            color = colors[idx + 1] if colors else _colors[1 + idx % len(_colors)]
            _kwargs = kwargs.copy()
            colors_other.append(_kwargs.setdefault('color', color))
            for band_idx in range(plotter.n_bands):
                for dist_idx, dists in enumerate(self_data['distances']):
                    xs = dists
                    ys = [other_data['frequency'][dist_idx][band_idx][j] * unit.factor for j in range(len(dists))]
                    ax.plot(xs, ys, **_kwargs | other_kwargs)
        color_self = ax.lines[0].get_color()
        ax.plot([], [], label=self._label or self_label, linewidth=2 * line_width, color=color_self)
        linestyle = other_kwargs.get('linestyle', '-')
        for color_other, label_other in zip(colors_other, other_plotter):
            ax.plot([], [], label=label_other, linewidth=2 * line_width, color=color_other, linestyle=linestyle)
            ax.legend(**legend_kwargs)
        return ax

    def plot_brillouin(self) -> None:
        """Plot the Brillouin zone."""
        q_pts = self._bs.qpoints
        labels = {q_pt.label: q_pt.frac_coords for q_pt in q_pts if q_pt.label}
        lines = [[q_pts[branch['start_index']].frac_coords, q_pts[branch['end_index']].frac_coords] for branch in self._bs.branches]
        plot_brillouin_zone(self._bs.lattice_rec, lines=lines, labels=labels)