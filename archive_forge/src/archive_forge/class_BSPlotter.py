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
class BSPlotter:
    """Class to plot or get data to facilitate the plot of band structure objects."""

    def __init__(self, bs: BandStructureSymmLine) -> None:
        """
        Args:
            bs: A BandStructureSymmLine object.
        """
        self._bs: list[BandStructureSymmLine] = []
        self._nb_bands: list[int] = []
        self.add_bs(bs)

    def _check_bs_kpath(self, bs_list: list[BandStructureSymmLine]) -> Literal[True]:
        """Helper method that check all the band objs in bs_list are
        BandStructureSymmLine objs and they all have the same kpath.
        """
        for bs in bs_list:
            if not isinstance(bs, BandStructureSymmLine):
                raise ValueError("BSPlotter only works with BandStructureSymmLine objects. A BandStructure object (on a uniform grid for instance and not along symmetry lines won't work)")
        if len(bs_list) == 1 and (not self._bs):
            return True
        if not self._bs:
            kpath_ref = [br['name'] for br in bs_list[0].branches]
        else:
            kpath_ref = [br['name'] for br in self._bs[0].branches]
        for bs in bs_list:
            if kpath_ref != [br['name'] for br in bs.branches]:
                msg = f'BSPlotter only works with BandStructureSymmLine which have the same kpath. \n{bs} has a different kpath!'
                raise ValueError(msg)
        return True

    def add_bs(self, bs: BandStructureSymmLine | list[BandStructureSymmLine]) -> None:
        """Method to add bands objects to the BSPlotter."""
        if not isinstance(bs, list):
            bs = [bs]
        if self._check_bs_kpath(bs):
            self._bs.extend(bs)
            self._nb_bands.extend([b.nb_bands for b in bs])

    def _make_ticks(self, ax: plt.Axes) -> plt.Axes:
        """Utility private method to add ticks to a band structure."""
        ticks = self.get_ticks()
        uniq_d = []
        uniq_l = []
        temp_ticks = list(zip(ticks['distance'], ticks['label']))
        for idx, t in enumerate(temp_ticks):
            if idx == 0:
                uniq_d.append(t[0])
                uniq_l.append(t[1])
                logger.debug(f'Adding label {t[0]} at {t[1]}')
            elif t[1] == temp_ticks[idx - 1][1]:
                logger.debug(f'Skipping label {t[1]}')
            else:
                logger.debug(f'Adding label {t[0]} at {t[1]}')
                uniq_d.append(t[0])
                uniq_l.append(t[1])
        logger.debug(f'Unique labels are {list(zip(uniq_d, uniq_l))}')
        ax.set_xticks(uniq_d)
        ax.set_xticklabels(uniq_l)
        for idx, label in enumerate(ticks['label']):
            if label is not None:
                if idx != 0:
                    if label == ticks['label'][idx - 1]:
                        logger.debug(f'already print label... skipping label {ticks['label'][idx]}')
                    else:
                        logger.debug(f'Adding a line at {ticks['distance'][idx]} for label {ticks['label'][idx]}')
                        ax.axvline(ticks['distance'][idx], color='k')
                else:
                    logger.debug(f'Adding a line at {ticks['distance'][idx]} for label {ticks['label'][idx]}')
                    ax.axvline(ticks['distance'][idx], color='k')
        return ax

    @staticmethod
    def _get_branch_steps(branches):
        """Method to find discontinuous branches."""
        steps = [0]
        for b1, b2 in zip(branches[:-1], branches[1:]):
            if b2['name'].split('-')[0] != b1['name'].split('-')[-1]:
                steps.append(b2['start_index'])
        steps.append(branches[-1]['end_index'] + 1)
        return steps

    @staticmethod
    def _rescale_distances(bs_ref, bs):
        """Method to rescale distances of bs to distances in bs_ref.
        This is used for plotting two bandstructures (same k-path)
        of different materials.
        """
        scaled_distances = []
        for br, br2 in zip(bs_ref.branches, bs.branches):
            start = br['start_index']
            end = br['end_index']
            max_d = bs_ref.distance[end]
            min_d = bs_ref.distance[start]
            s2 = br2['start_index']
            e2 = br2['end_index']
            np = e2 - s2
            if np == 0:
                scaled_distances.extend([min_d])
            else:
                scaled_distances.extend([(max_d - min_d) / np * i + min_d for i in range(np + 1)])
        return scaled_distances

    def bs_plot_data(self, zero_to_efermi=True, bs=None, bs_ref=None, split_branches=True):
        """Get the data nicely formatted for a plot.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            bs: the bandstructure to get the data from. If not provided, the first
                one in the self._bs list will be used.
            bs_ref: is the bandstructure of reference when a rescale of the distances
                is need to plot multiple bands
            split_branches: if True distances and energies are split according to the
                branches. If False distances and energies are split only where branches
                are discontinuous (reducing the number of lines to plot).

        Returns:
            dict: A dictionary of the following format:
            ticks: A dict with the 'distances' at which there is a kpoint (the
            x axis) and the labels (None if no label).
            energy: A dict storing bands for spin up and spin down data
            {Spin:[np.array(nb_bands,kpoints),...]} as a list of discontinuous kpath
            of energies. The energy of multiple continuous branches are stored together.
            vbm: A list of tuples (distance,energy) marking the vbms. The
            energies are shifted with respect to the Fermi level is the
            option has been selected.
            cbm: A list of tuples (distance,energy) marking the cbms. The
            energies are shifted with respect to the Fermi level is the
            option has been selected.
            lattice: The reciprocal lattice.
            zero_energy: This is the energy used as zero for the plot.
            band_gap:A string indicating the band gap and its nature (empty if
            it's a metal).
            is_metal: True if the band structure is metallic (i.e., there is at
            least one band crossing the Fermi level).
        """
        if bs is None:
            bs = self._bs[0] if isinstance(self._bs, list) else self._bs
        energies = {str(sp): [] for sp in bs.bands}
        bs_is_metal = bs.is_metal()
        if not bs_is_metal:
            vbm = bs.get_vbm()
            cbm = bs.get_cbm()
        zero_energy = 0.0
        if zero_to_efermi:
            zero_energy = bs.efermi if bs_is_metal else vbm['energy']
        distances = bs.distance
        if bs_ref is not None and bs_ref.branches != bs.branches:
            distances = self._rescale_distances(bs_ref, bs)
        if split_branches:
            steps = [br['end_index'] + 1 for br in bs.branches][:-1]
        else:
            steps = self._get_branch_steps(bs.branches)[1:-1]
        distances = np.split(distances, steps)
        for sp in bs.bands:
            energies[str(sp)] = np.hsplit(bs.bands[sp] - zero_energy, steps)
        ticks = self.get_ticks()
        vbm_plot = []
        cbm_plot = []
        bg_str = ''
        if not bs_is_metal:
            for index in cbm['kpoint_index']:
                cbm_plot.append((bs.distance[index], cbm['energy'] - zero_energy if zero_to_efermi else cbm['energy']))
            for index in vbm['kpoint_index']:
                vbm_plot.append((bs.distance[index], vbm['energy'] - zero_energy if zero_to_efermi else vbm['energy']))
            bg = bs.get_band_gap()
            direct = 'Indirect'
            if bg['direct']:
                direct = 'Direct'
            bg_str = f'{direct} {bg['transition']} bandgap = {bg['energy']}'
        return {'ticks': ticks, 'distances': distances, 'energy': energies, 'vbm': vbm_plot, 'cbm': cbm_plot, 'lattice': bs.lattice_rec.as_dict(), 'zero_energy': zero_energy, 'is_metal': bs_is_metal, 'band_gap': bg_str}

    @staticmethod
    def _interpolate_bands(distances, energies, smooth_tol=0, smooth_k=3, smooth_np=100):
        """Method that interpolates the provided energies using B-splines as
        implemented in scipy.interpolate. Distances and energies has to provided
        already split into pieces (branches work good, for longer segments
        the interpolation may fail).

        Interpolation failure can be caused by trying to fit an entire
        band with one spline rather than fitting with piecewise splines
        (splines are ill-suited to fit discontinuities).

        The number of splines used to fit a band is determined by the
        number of branches (high symmetry lines) defined in the
        BandStructureSymmLine object (see BandStructureSymmLine._branches).
        """
        int_energies, int_distances = ([], [])
        smooth_k_orig = smooth_k
        for dist, ene in zip(distances, energies):
            br_en = []
            warning_nan = f'WARNING! Distance / branch, band cannot be interpolated. See full warning in source. If this is not a mistake, try increasing smooth_tol. Current smooth_tol={smooth_tol!r}.'
            warning_m_fewer_k = f'The number of points (m) has to be higher then the order (k) of the splines. In this branch {len(dist)} points are found, while k is set to {smooth_k}. Smooth_k will be reduced to {smooth_k - 1} for this branch.'
            if len(dist) in (2, 3):
                smooth_k = len(dist) - 1
                warnings.warn(warning_m_fewer_k)
            elif len(dist) == 1:
                warnings.warn('Skipping single point branch')
                continue
            int_distances.append(np.linspace(dist[0], dist[-1], smooth_np))
            for ien in ene:
                tck = scint.splrep(dist, ien, s=smooth_tol, k=smooth_k)
                br_en.append(scint.splev(int_distances[-1], tck))
            smooth_k = smooth_k_orig
            int_energies.append(np.vstack(br_en))
            if np.any(np.isnan(int_energies[-1])):
                warnings.warn(warning_nan)
        return (int_distances, int_energies)

    def get_plot(self, zero_to_efermi=True, ylim=None, smooth=False, vbm_cbm_marker=False, smooth_tol=0, smooth_k=3, smooth_np=100, bs_labels=None):
        """Get a matplotlib object for the bandstructures plot.
        Multiple bandstructure objs are plotted together if they have the
        same high symm path.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            ylim: Specify the y-axis (energy) limits; by default None let
                the code choose. It is vbm-4 and cbm+4 if insulator
                efermi-10 and efermi+10 if metal
            smooth (bool or list(bools)): interpolates the bands by a spline cubic.
                A single bool values means to interpolate all the bandstructure objs.
                A list of bools allows to select the bandstructure obs to interpolate.
            vbm_cbm_marker (bool): if True, a marker is added to the vbm and cbm.
            smooth_tol (float) : tolerance for fitting spline to band data.
                Default is None such that no tolerance will be used.
            smooth_k (int): degree of splines 1<k<5
            smooth_np (int): number of interpolated points per each branch.
            bs_labels: labels for each band for the plot legend.
        """
        ax = pretty_plot(12, 8)
        if isinstance(smooth, bool):
            smooth = [smooth] * len(self._bs)
        handles = []
        vbm_min, cbm_max = ([], [])
        colors = next(iter(plt.rcParams['axes.prop_cycle'].by_key().values()))
        for ibs, bs in enumerate(self._bs):
            bs_ref = self._bs[0] if len(self._bs) > 1 and ibs > 0 else None
            if smooth[ibs]:
                data = self.bs_plot_data(zero_to_efermi, bs, bs_ref, split_branches=True)
            else:
                data = self.bs_plot_data(zero_to_efermi, bs, bs_ref, split_branches=False)
            one_is_metal = False
            if not one_is_metal and data['is_metal']:
                one_is_metal = data['is_metal']
            if not data['is_metal']:
                cbm_max.append(data['cbm'][0][1])
                vbm_min.append(data['vbm'][0][1])
            else:
                cbm_max.append(bs.efermi)
                vbm_min.append(bs.efermi)
            for sp in bs.bands:
                ls = '-' if str(sp) == '1' else '--'
                bs_label = f'Band {ibs} {sp.name}' if bs_labels is None else f'{bs_labels[ibs]} {sp.name}'
                handles.append(mlines.Line2D([], [], lw=2, ls=ls, color=colors[ibs], label=bs_label))
                distances, energies = (data['distances'], data['energy'][str(sp)])
                if smooth[ibs]:
                    distances, energies = self._interpolate_bands(distances, energies, smooth_tol=smooth_tol, smooth_k=smooth_k, smooth_np=smooth_np)
                    distances = np.hstack(distances)
                    energies = np.hstack(energies)
                    steps = self._get_branch_steps(bs.branches)[1:-1]
                    distances = np.split(distances, steps)
                    energies = np.hsplit(energies, steps)
                for dist, ene in zip(distances, energies):
                    ax.plot(dist, ene.T, c=colors[ibs], ls=ls)
            if vbm_cbm_marker:
                for cbm in data['cbm']:
                    ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                for vbm in data['vbm']:
                    ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
            if not zero_to_efermi:
                ef = bs.efermi
                ax.axhline(ef, lw=2, ls='-.', color=colors[ibs])
        e_min = -4
        e_max = 4
        if one_is_metal:
            e_min = -10
            e_max = 10
        if ylim is None:
            if zero_to_efermi:
                ax.set_ylim(e_min, e_max if one_is_metal else max(cbm_max) + e_max)
            else:
                all_efermi = [b.efermi for b in self._bs]
                ll = min([min(vbm_min), min(all_efermi)])
                hh = max([max(cbm_max), max(all_efermi)])
                ax.set_ylim(ll + e_min, hh + e_max)
        else:
            ax.set_ylim(ylim)
        self._make_ticks(ax)
        ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=30)
        ylabel = '$\\mathrm{E\\ -\\ E_f\\ (eV)}$' if zero_to_efermi else '$\\mathrm{Energy\\ (eV)}$'
        ax.set_ylabel(ylabel, fontsize=30)
        x_max = data['distances'][-1][-1]
        ax.set_xlim(0, x_max)
        ax.legend(handles=handles)
        plt.tight_layout()

        def fix_layout(event) -> None:
            if event.name == 'key_press_event' and event.key == 't' or event.name == 'resize_event':
                plt.tight_layout()
                plt.gcf().canvas.draw()
        ax.figure.canvas.mpl_connect('key_press_event', fix_layout)
        ax.figure.canvas.mpl_connect('resize_event', fix_layout)
        return ax

    def show(self, zero_to_efermi=True, ylim=None, smooth=False, smooth_tol=None) -> None:
        """Show the plot using matplotlib.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            ylim: Specify the y-axis (energy) limits; by default None let
                the code choose. It is vbm-4 and cbm+4 if insulator
                efermi-10 and efermi+10 if metal
            smooth: interpolates the bands by a spline cubic
            smooth_tol (float) : tolerance for fitting spline to band data.
                Default is None such that no tolerance will be used.
        """
        self.get_plot(zero_to_efermi, ylim, smooth)
        plt.show()

    def save_plot(self, filename: str, ylim=None, zero_to_efermi=True, smooth=False) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename (str): Filename to write to. Must include extension to specify image format.
            ylim: Specifies the y-axis limits.
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            smooth: Cubic spline interpolation of the bands.
        """
        self.get_plot(ylim=ylim, zero_to_efermi=zero_to_efermi, smooth=smooth)
        plt.savefig(filename)
        plt.close()

    def get_ticks(self):
        """Get all ticks and labels for a band structure plot.

        Returns:
            dict: A dictionary with 'distance': a list of distance at which
            ticks should be set and 'label': a list of label for each of those
            ticks.
        """
        bs = self._bs[0] if isinstance(self._bs, list) else self._bs
        ticks, distance = ([], [])
        for br in bs.branches:
            start, end = (br['start_index'], br['end_index'])
            labels = br['name'].split('-')
            if labels[0] == labels[1]:
                continue
            for idx, label in enumerate(labels):
                if label.startswith('\\') or '_' in label:
                    labels[idx] = f'${label}$'
            if ticks and labels[0] != ticks[-1]:
                ticks[-1] += f'$\\mid${labels[0]}'
                ticks.append(labels[1])
                distance.append(bs.distance[end])
            else:
                ticks.extend(labels)
                distance.extend([bs.distance[start], bs.distance[end]])
        return {'distance': distance, 'label': ticks}

    def get_ticks_old(self):
        """Get all ticks and labels for a band structure plot.

        Returns:
            dict: A dictionary with 'distance': a list of distance at which
            ticks should be set and 'label': a list of label for each of those
            ticks.
        """
        bs = self._bs[0]
        tick_distance = []
        tick_labels = []
        previous_label = bs.kpoints[0].label
        previous_branch = bs.branches[0]['name']
        for idx, kpt in enumerate(bs.kpoints):
            if kpt.label is not None:
                tick_distance.append(bs.distance[idx])
                this_branch = None
                for b in bs.branches:
                    if b['start_index'] <= idx <= b['end_index']:
                        this_branch = b['name']
                        break
                if kpt.label != previous_label and previous_branch != this_branch:
                    label1 = kpt.label
                    if label1.startswith('\\') or label1.find('_') != -1:
                        label1 = f'${label1}$'
                    label0 = previous_label
                    if label0.startswith('\\') or label0.find('_') != -1:
                        label0 = f'${label0}$'
                    tick_labels.pop()
                    tick_distance.pop()
                    tick_labels.append(label0 + '$\\mid$' + label1)
                elif kpt.label.startswith('\\') or kpt.label.find('_') != -1:
                    tick_labels.append(f'${kpt.label}$')
                else:
                    tick_labels.append(kpt.label)
                previous_label = kpt.label
                previous_branch = this_branch
        return {'distance': tick_distance, 'label': tick_labels}

    def plot_compare(self, other_plotter, legend=True) -> plt.Axes:
        """Plot two band structure for comparison. One is in red the other in blue
        (no difference in spins). The two band structures need to be defined
        on the same symmetry lines! and the distance between symmetry lines is
        the one of the band structure used to build the BSPlotter.

        Args:
            other_plotter: Another band structure object defined along the same symmetry lines
            legend: True to add a legend to the plot

        Returns:
            plt.Axes: matplotlib Axes object with both band structures
        """
        warnings.warn('Deprecated method. Use BSPlotter([sbs1,sbs2,...]).get_plot() instead.')
        ax = self.get_plot()
        data_orig = self.bs_plot_data()
        data = other_plotter.bs_plot_data()
        band_linewidth = 1
        for i in range(other_plotter._nb_bands):
            for d in range(len(data_orig['distances'])):
                ax.plot(data_orig['distances'][d], [e[str(Spin.up)][i] for e in data['energy']][d], 'c-', linewidth=band_linewidth)
                if other_plotter._bs.is_spin_polarized:
                    ax.plot(data_orig['distances'][d], [e[str(Spin.down)][i] for e in data['energy']][d], 'm--', linewidth=band_linewidth)
        if legend:
            handles = [mlines.Line2D([], [], linewidth=2, color='b', label='bs 1 up'), mlines.Line2D([], [], linewidth=2, color='r', label='bs 1 down', linestyle='--'), mlines.Line2D([], [], linewidth=2, color='c', label='bs 2 up'), mlines.Line2D([], [], linewidth=2, color='m', linestyle='--', label='bs 2 down')]
            ax.legend(handles=handles)
        return ax

    def plot_brillouin(self):
        """Plot the Brillouin zone.

        Returns:
            plt.Figure: A matplotlib figure object with the Brillouin zone.
        """
        labels = {}
        for k in self._bs[0].kpoints:
            if k.label:
                labels[k.label] = k.frac_coords
        lines = []
        for branch in self._bs[0].branches:
            kpts = self._bs[0].kpoints
            start_idx, end_idx = (branch['start_index'], branch['end_index'])
            lines.append([kpts[start_idx].frac_coords, kpts[end_idx].frac_coords])
        return plot_brillouin_zone(self._bs[0].lattice_rec, lines=lines, labels=labels)