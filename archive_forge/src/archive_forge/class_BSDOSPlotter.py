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
class BSDOSPlotter:
    """A joint, aligned band structure and density of states plot. Contributions
    from Jan Pohls as well as the online example from Germain Salvato-Vallverdu:
    http://gvallver.perso.univ-pau.fr/?p=587.
    """

    def __init__(self, bs_projection: Literal['elements'] | None='elements', dos_projection: str='elements', vb_energy_range: float=4, cb_energy_range: float=4, fixed_cb_energy: bool=False, egrid_interval: float=1, font: str='Times New Roman', axis_fontsize: float=20, tick_fontsize: float=15, legend_fontsize: float=14, bs_legend: str='best', dos_legend: str='best', rgb_legend: bool=True, fig_size: tuple[float, float]=(11, 8.5)) -> None:
        """Instantiate plotter settings.

        Args:
            bs_projection ('elements' | None): Whether to project the bands onto elements.
            dos_projection (str): "elements", "orbitals", or None
            vb_energy_range (float): energy in eV to show of valence bands
            cb_energy_range (float): energy in eV to show of conduction bands
            fixed_cb_energy (bool): If true, the cb_energy_range will be interpreted
                as constant (i.e., no gap correction for cb energy)
            egrid_interval (float): interval for grid marks
            font (str): font family
            axis_fontsize (float): font size for axis
            tick_fontsize (float): font size for axis tick labels
            legend_fontsize (float): font size for legends
            bs_legend (str): matplotlib string location for legend or None
            dos_legend (str): matplotlib string location for legend or None
            rgb_legend (bool): (T/F) whether to draw RGB triangle/bar for element proj.
            fig_size(tuple): dimensions of figure size (width, height)
        """
        self.bs_projection = bs_projection
        self.dos_projection = dos_projection
        self.vb_energy_range = vb_energy_range
        self.cb_energy_range = cb_energy_range
        self.fixed_cb_energy = fixed_cb_energy
        self.egrid_interval = egrid_interval
        self.font = font
        self.axis_fontsize = axis_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.bs_legend = bs_legend
        self.dos_legend = dos_legend
        self.rgb_legend = rgb_legend
        self.fig_size = fig_size

    def get_plot(self, bs: BandStructureSymmLine, dos: Dos | CompleteDos | None=None) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
        """Get a matplotlib plot object.

        Args:
            bs (BandStructureSymmLine): the bandstructure to plot. Projection
                data must exist for projected plots.
            dos (Dos): the Dos to plot. Projection data must exist (i.e.,
                CompleteDos) for projected plots.

        Returns:
            plt.Axes | tuple[plt.Axes, plt.Axes]: matplotlib axes for the band structure and DOS, resp.
        """
        bs_projection = self.bs_projection
        if dos:
            elements = [e.symbol for e in dos.structure.elements]
        elif bs_projection and bs.structure:
            elements = [e.symbol for e in bs.structure.elements]
        else:
            elements = []
        rgb_legend = self.rgb_legend and bs_projection and (bs_projection.lower() == 'elements') and (len(elements) in [2, 3, 4])
        if bs_projection and bs_projection.lower() == 'elements' and (len(elements) not in [2, 3, 4] or not bs.get_projection_on_elements()):
            warnings.warn("Cannot get element projected data; either the projection data doesn't exist, or you don't have a compound with exactly 2 or 3 or 4 unique elements.")
            bs_projection = None
        emin = -self.vb_energy_range
        emax = self.cb_energy_range if self.fixed_cb_energy else self.cb_energy_range + bs.get_band_gap()['energy']
        xlabels = []
        xlabel_distances = []
        x_distances_list = []
        prev_right_klabel = None
        for branch in bs.branches:
            x_distances = []
            left_k, right_k = branch['name'].split('-')
            if left_k[0] == '\\' or '_' in left_k:
                left_k = f'${left_k}$'
            if right_k[0] == '\\' or '_' in right_k:
                right_k = f'${right_k}$'
            if prev_right_klabel is None:
                xlabels.append(left_k)
                xlabel_distances.append(0)
            elif prev_right_klabel != left_k:
                xlabels[-1] = f'{xlabels[-1]}$\\mid$ {left_k}'
            xlabels.append(right_k)
            prev_right_klabel = right_k
            left_kpoint = bs.kpoints[branch['start_index']].cart_coords
            right_kpoint = bs.kpoints[branch['end_index']].cart_coords
            distance = np.linalg.norm(right_kpoint - left_kpoint)
            xlabel_distances.append(xlabel_distances[-1] + distance)
            npts = branch['end_index'] - branch['start_index']
            distance_interval = distance / npts
            x_distances.append(xlabel_distances[-2])
            for _ in range(npts):
                x_distances.append(x_distances[-1] + distance_interval)
            x_distances_list.append(x_distances)
        gs = GridSpec(1, 2, width_ratios=[2, 1]) if dos else GridSpec(1, 1)
        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor('white')
        bs_ax = plt.subplot(gs[0])
        if dos:
            dos_ax = plt.subplot(gs[1])
        bs_ax.set_xlim(0, x_distances_list[-1][-1])
        bs_ax.set_ylim(emin, emax)
        if dos:
            dos_ax.set_ylim(emin, emax)
        bs_ax.set_xticks(xlabel_distances)
        bs_ax.set_xticklabels(xlabels, size=self.tick_fontsize)
        bs_ax.set_xlabel('Wavevector $k$', fontsize=self.axis_fontsize, family=self.font)
        bs_ax.set_ylabel('$E-E_F$ / eV', fontsize=self.axis_fontsize, family=self.font)
        bs_ax.hlines(y=0, xmin=0, xmax=x_distances_list[-1][-1], color='k', lw=2)
        bs_ax.set_yticks(np.arange(emin, emax + 1e-05, self.egrid_interval))
        bs_ax.set_yticklabels(np.arange(emin, emax + 1e-05, self.egrid_interval), size=self.tick_fontsize)
        bs_ax.set_axisbelow(b=True)
        bs_ax.grid(color=[0.5, 0.5, 0.5], linestyle='dotted', linewidth=1)
        if dos:
            dos_ax.set_yticks(np.arange(emin, emax + 1e-05, self.egrid_interval))
            dos_ax.set_yticklabels([])
            dos_ax.grid(color=[0.5, 0.5, 0.5], linestyle='dotted', linewidth=1)
        band_energies: dict[Spin, list[float]] = {}
        for spin in (Spin.up, Spin.down):
            if spin in bs.bands:
                band_energies[spin] = []
                for band in bs.bands[spin]:
                    band = cast(list[float], band)
                    band_energies[spin].append([e - bs.efermi for e in band])
        if dos:
            dos_energies = [e - dos.efermi for e in dos.energies]
        colordata = self._get_colordata(bs, elements, bs_projection)
        for spin in (Spin.up, Spin.down):
            if spin in band_energies:
                linestyles = 'solid' if spin == Spin.up else 'dotted'
                for band_idx, band in enumerate(band_energies[spin]):
                    current_pos = 0
                    for x_distances in x_distances_list:
                        sub_band = band[current_pos:current_pos + len(x_distances)]
                        self._rgbline(bs_ax, x_distances, sub_band, colordata[spin][band_idx, :, 0][current_pos:current_pos + len(x_distances)], colordata[spin][band_idx, :, 1][current_pos:current_pos + len(x_distances)], colordata[spin][band_idx, :, 2][current_pos:current_pos + len(x_distances)], linestyles=linestyles)
                        current_pos += len(x_distances)
        if dos:
            for spin in (Spin.up, Spin.down):
                if spin in dos.densities:
                    dos_densities = dos.densities[spin] * int(spin)
                    label = 'total' if spin == Spin.up else None
                    dos_ax.plot(dos_densities, dos_energies, color=(0.6, 0.6, 0.6), label=label)
                    dos_ax.fill_betweenx(dos_energies, 0, dos_densities, color=(0.7, 0.7, 0.7), facecolor=(0.7, 0.7, 0.7))
                    if self.dos_projection is None:
                        pass
                    elif self.dos_projection.lower() == 'elements':
                        colors = ['b', 'r', 'g', 'm', 'y', 'c', 'k', 'w']
                        el_dos = dos.get_element_dos()
                        for idx, el in enumerate(elements):
                            dos_densities = el_dos[Element(el)].densities[spin] * int(spin)
                            label = el if spin == Spin.up else None
                            dos_ax.plot(dos_densities, dos_energies, color=colors[idx], label=label)
                    elif self.dos_projection.lower() == 'orbitals':
                        colors = ['b', 'r', 'g', 'm']
                        spd_dos = dos.get_spd_dos()
                        for idx, orb in enumerate([OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]):
                            if orb in spd_dos:
                                dos_densities = spd_dos[orb].densities[spin] * int(spin)
                                label = orb if spin == Spin.up else None
                                dos_ax.plot(dos_densities, dos_energies, color=colors[idx], label=label)
            emin_idx = next((x[0] for x in enumerate(dos_energies) if x[1] >= emin))
            emax_idx = len(dos_energies) - next((x[0] for x in enumerate(reversed(dos_energies)) if x[1] <= emax))
            dos_xmin = 0 if Spin.down not in dos.densities else -max(dos.densities[Spin.down][emin_idx:emax_idx + 1] * 1.05)
            dos_xmax = max([max(dos.densities[Spin.up][emin_idx:emax_idx]) * 1.05, abs(dos_xmin)])
            dos_ax.set_xlim(dos_xmin, dos_xmax)
            dos_ax.set_xticklabels([])
            dos_ax.hlines(y=0, xmin=dos_xmin, xmax=dos_xmax, color='k', lw=2)
            dos_ax.set_xlabel('DOS', fontsize=self.axis_fontsize, family=self.font)
        if self.bs_legend and (not rgb_legend):
            handles = []
            if bs_projection is None:
                handles = [mlines.Line2D([], [], linewidth=2, color='k', label='spin up'), mlines.Line2D([], [], linewidth=2, color='b', linestyle='dotted', label='spin down')]
            elif bs_projection.lower() == 'elements':
                colors = ['b', 'r', 'g']
                for idx, el in enumerate(elements):
                    handles.append(mlines.Line2D([], [], linewidth=2, color=colors[idx], label=el))
            bs_ax.legend(handles=handles, fancybox=True, prop={'size': self.legend_fontsize, 'family': self.font}, loc=self.bs_legend)
        elif self.bs_legend and rgb_legend:
            if len(elements) == 2:
                self._rb_line(bs_ax, elements[1], elements[0], loc=self.bs_legend)
            elif len(elements) == 3:
                self._rgb_triangle(bs_ax, elements[1], elements[2], elements[0], loc=self.bs_legend)
            elif len(elements) == 4:
                self._cmyk_triangle(bs_ax, elements[1], elements[2], elements[0], elements[3], loc=self.bs_legend)
        if dos and self.dos_legend:
            dos_ax.legend(fancybox=True, prop={'size': self.legend_fontsize, 'family': self.font}, loc=self.dos_legend)
        plt.subplots_adjust(wspace=0.1)
        if dos:
            return (bs_ax, dos_ax)
        return bs_ax

    @staticmethod
    def _rgbline(ax, k, e, red, green, blue, alpha=1, linestyles='solid') -> None:
        """An RGB colored line for plotting.
        creation of segments based on:
        http://nbviewer.ipython.org/urls/raw.github.com/dpsanders/matplotlib-examples/master/colorline.ipynb.

        Args:
            ax: matplotlib axis
            k: x-axis data (k-points)
            e: y-axis data (energies)
            red: red data
            green: green data
            blue: blue data
            alpha: alpha values data
            linestyles: linestyle for plot (e.g., "solid" or "dotted").
        """
        pts = np.array([k, e]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        n_seg = len(k) - 1
        red = [0.5 * (red[i] + red[i + 1]) for i in range(n_seg)]
        green = [0.5 * (green[i] + green[i + 1]) for i in range(n_seg)]
        blue = [0.5 * (blue[i] + blue[i + 1]) for i in range(n_seg)]
        alpha = np.ones(n_seg, float) * alpha
        lc = LineCollection(seg, colors=list(zip(red, green, blue, alpha)), linewidth=2, linestyles=linestyles)
        ax.add_collection(lc)

    @staticmethod
    def _get_colordata(bs, elements, bs_projection):
        """Get color data, including projected band structures.

        Args:
            bs: Bandstructure object
            elements: elements (in desired order) for setting to blue, red, green
            bs_projection: None for no projection, "elements" for element projection

        Returns:
            Dictionary representation of color data.
        """
        contribs = {}
        if bs_projection and bs_projection.lower() == 'elements':
            projections = bs.get_projection_on_elements()
        for spin in (Spin.up, Spin.down):
            if spin in bs.bands:
                contribs[spin] = []
                for band_idx in range(bs.nb_bands):
                    colors = []
                    for k_idx in range(len(bs.kpoints)):
                        if bs_projection and bs_projection.lower() == 'elements':
                            c = [0, 0, 0, 0]
                            projs = projections[spin][band_idx][k_idx]
                            projs = {k: v ** 2 for k, v in projs.items()}
                            total = sum(projs.values())
                            if total > 0:
                                for idx, e in enumerate(elements):
                                    c[idx] = math.sqrt(projs[e] / total)
                            c = [c[1], c[2], c[0], c[3]]
                            if len(elements) == 4:
                                c = [(1 - c[0]) * (1 - c[3]), (1 - c[1]) * (1 - c[3]), (1 - c[2]) * (1 - c[3])]
                            else:
                                c = [c[0], c[1], c[2]]
                        else:
                            c = [0, 0, 0] if spin == Spin.up else [0, 0, 1]
                        colors.append(c)
                    contribs[spin].append(colors)
                contribs[spin] = np.array(contribs[spin])
        return contribs

    @staticmethod
    def _cmyk_triangle(ax, c_label, m_label, y_label, k_label, loc) -> None:
        """Draw an RGB triangle legend on the desired axis."""
        if loc not in range(1, 11):
            loc = 2
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width=1.5, height=1.5, loc=loc)
        mesh = 35
        x = []
        y = []
        color = []
        for c in range(mesh):
            for ye in range(mesh):
                for m in range(mesh):
                    if not (c == mesh - 1 and ye == mesh - 1 and (m == mesh - 1)) and (not (c == 0 and ye == 0 and (m == 0))):
                        c1 = c / (c + ye + m)
                        ye1 = ye / (c + ye + m)
                        m1 = m / (c + ye + m)
                        x.append(0.33 * (2.0 * ye1 + c1) / (c1 + ye1 + m1))
                        y.append(0.33 * np.sqrt(3) * c1 / (c1 + ye1 + m1))
                        rc = 1 - c / (mesh - 1)
                        gc = 1 - m / (mesh - 1)
                        bc = 1 - ye / (mesh - 1)
                        color.append([rc, gc, bc])
        inset_ax.scatter(x, y, s=7, marker='.', edgecolor=color)
        inset_ax.set_xlim([-0.35, 1.0])
        inset_ax.set_ylim([-0.35, 1.0])
        common = dict(fontsize=13, family='Times New Roman')
        inset_ax.text(0.7, -0.2, m_label, **common, color=(0, 0, 0), horizontalalignment='left')
        inset_ax.text(0.325, 0.7, c_label, **common, color=(0, 0, 0), horizontalalignment='center')
        inset_ax.text(-0.05, -0.2, y_label, **common, color=(0, 0, 0), horizontalalignment='right')
        inset_ax.text(0.325, 0.22, k_label, **common, color=(1, 1, 1), horizontalalignment='center')
        inset_ax.axis('off')

    @staticmethod
    def _rgb_triangle(ax, r_label, g_label, b_label, loc) -> None:
        """Draw an RGB triangle legend on the desired axis."""
        if loc not in range(1, 11):
            loc = 2
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width=1, height=1, loc=loc)
        mesh = 35
        x = []
        y = []
        color = []
        for r in range(mesh):
            for g in range(mesh):
                for b in range(mesh):
                    if not (r == 0 and b == 0 and (g == 0)):
                        r1 = r / (r + g + b)
                        g1 = g / (r + g + b)
                        b1 = b / (r + g + b)
                        x.append(0.33 * (2.0 * g1 + r1) / (r1 + b1 + g1))
                        y.append(0.33 * np.sqrt(3) * r1 / (r1 + b1 + g1))
                        rc = math.sqrt(r ** 2 / (r ** 2 + g ** 2 + b ** 2))
                        gc = math.sqrt(g ** 2 / (r ** 2 + g ** 2 + b ** 2))
                        bc = math.sqrt(b ** 2 / (r ** 2 + g ** 2 + b ** 2))
                        color.append([rc, gc, bc])
        inset_ax.scatter(x, y, s=7, marker='.', edgecolor=color)
        inset_ax.set_xlim([-0.35, 1.0])
        inset_ax.set_ylim([-0.35, 1.0])
        inset_ax.text(0.7, -0.2, g_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='left')
        inset_ax.text(0.325, 0.7, r_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='center')
        inset_ax.text(-0.05, -0.2, b_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='right')
        inset_ax.axis('off')

    @staticmethod
    def _rb_line(ax, r_label, b_label, loc) -> None:
        if loc not in range(1, 11):
            loc = 2
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width=1.2, height=0.4, loc=loc)
        x, y, color = ([], [], [])
        for idx in range(1000):
            x.append(idx / 1800.0 + 0.55)
            y.append(0)
            color.append([math.sqrt(c) for c in [1 - (idx / 1000) ** 2, 0, (idx / 1000) ** 2]])
        inset_ax.scatter(x, y, s=250.0, marker='s', c=color)
        inset_ax.set_xlim([-0.1, 1.7])
        inset_ax.text(1.35, 0, b_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='left', verticalalignment='center')
        inset_ax.text(0.3, 0, r_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='right', verticalalignment='center')
        inset_ax.axis('off')