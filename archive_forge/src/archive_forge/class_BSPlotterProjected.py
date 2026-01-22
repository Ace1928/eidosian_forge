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
class BSPlotterProjected(BSPlotter):
    """Class to plot or get data to facilitate the plot of band structure objects
    projected along orbitals, elements or sites.
    """

    def __init__(self, bs) -> None:
        """
        Args:
            bs: A BandStructureSymmLine object with projections.
        """
        if isinstance(bs, list):
            warnings.warn('Multiple bands are not handled by BSPlotterProjected. The first band in the list will be considered')
            bs = bs[0]
        if len(bs.projections) == 0:
            raise ValueError('try to plot projections on a band structure without any')
        self._bs = bs
        self._nb_bands = bs.nb_bands

    def _get_projections_by_branches(self, dictio):
        proj = self._bs.get_projections_on_elements_and_orbitals(dictio)
        proj_br = []
        for b in self._bs.branches:
            if self._bs.is_spin_polarized:
                proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)], str(Spin.down): [[] for _ in range(self._nb_bands)]})
            else:
                proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)]})
            for i in range(self._nb_bands):
                for j in range(b['start_index'], b['end_index'] + 1):
                    proj_br[-1][str(Spin.up)][i].append({e: {o: proj[Spin.up][i][j][e][o] for o in proj[Spin.up][i][j][e]} for e in proj[Spin.up][i][j]})
            if self._bs.is_spin_polarized:
                for b in self._bs.branches:
                    for i in range(self._nb_bands):
                        for j in range(b['start_index'], b['end_index'] + 1):
                            proj_br[-1][str(Spin.down)][i].append({e: {o: proj[Spin.down][i][j][e][o] for o in proj[Spin.down][i][j][e]} for e in proj[Spin.down][i][j]})
        return proj_br

    def get_projected_plots_dots(self, dictio, zero_to_efermi=True, ylim=None, vbm_cbm_marker=False):
        """Method returning a plot composed of subplots along different elements
        and orbitals.

        Args:
            dictio: The element and orbitals you want a projection on. The
                format is {Element:[Orbitals]} for instance
                {'Cu':['d','s'],'O':['p']} will give projections for Cu on
                d and s orbitals and on oxygen p.
                If you use this class to plot LobsterBandStructureSymmLine,
                the orbitals are named as in the FATBAND filename, e.g.
                "2p" or "2p_x"
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            ylim: Specify the y-axis limits. Defaults to None.
            vbm_cbm_marker: Add markers for the VBM and CBM. Defaults to False.

        Returns:
            list[plt.Axes]: A list with different subfigures for each projection
            The blue and red colors are for spin up and spin down.
            The bigger the red or blue dot in the band structure the higher
            character for the corresponding element and orbital.
        """
        band_linewidth = 1.0
        fig_cols = len(dictio) * 100
        fig_rows = max((len(v) for v in dictio.values())) * 10
        proj = self._get_projections_by_branches(dictio)
        data = self.bs_plot_data(zero_to_efermi)
        ax = pretty_plot(12, 8)
        e_min = -4
        e_max = 4
        if self._bs.is_metal():
            e_min = -10
            e_max = 10
        for el in dictio:
            for idx, key in enumerate(dictio[el], 1):
                ax = plt.subplot(fig_rows + fig_cols + idx)
                self._make_ticks(ax)
                for b in range(len(data['distances'])):
                    for i in range(self._nb_bands):
                        ax.plot(data['distances'][b], data['energy'][str(Spin.up)][b][i], 'b-', linewidth=band_linewidth)
                        if self._bs.is_spin_polarized:
                            ax.plot(data['distances'][b], data['energy'][str(Spin.down)][b][i], 'r--', linewidth=band_linewidth)
                            for j in range(len(data['energy'][str(Spin.up)][b][i])):
                                ax.plot(data['distances'][b][j], data['energy'][str(Spin.down)][b][i][j], 'ro', markersize=proj[b][str(Spin.down)][i][j][str(el)][key] * 15.0)
                        for j in range(len(data['energy'][str(Spin.up)][b][i])):
                            ax.plot(data['distances'][b][j], data['energy'][str(Spin.up)][b][i][j], 'bo', markersize=proj[b][str(Spin.up)][i][j][str(el)][key] * 15.0)
                if ylim is None:
                    if self._bs.is_metal():
                        if zero_to_efermi:
                            ax.set_ylim(e_min, e_max)
                        else:
                            ax.set_ylim(self._bs.efermi + e_min, self._bs.efermi + e_max)
                    else:
                        if vbm_cbm_marker:
                            for cbm in data['cbm']:
                                ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                            for vbm in data['vbm']:
                                ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
                        ax.set_ylim(data['vbm'][0][1] + e_min, data['cbm'][0][1] + e_max)
                else:
                    ax.set_ylim(ylim)
                ax.set_title(f'{el} {key}')
        return plt.gcf().axes

    @no_type_check
    def get_elt_projected_plots(self, zero_to_efermi: bool=True, ylim=None, vbm_cbm_marker: bool=False) -> plt.Axes:
        """Method returning a plot composed of subplots along different elements.

        Returns:
            np.ndarray[plt.Axes]: 2x2 array of plt.Axes with different subfigures for each projection
                The blue and red colors are for spin up and spin down
                The bigger the red or blue dot in the band structure the higher
                character for the corresponding element and orbital
        """
        band_linewidth = 1.0
        proj = self._get_projections_by_branches({e.symbol: ['s', 'p', 'd'] for e in self._bs.structure.elements})
        data = self.bs_plot_data(zero_to_efermi)
        _fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax = pretty_plot(12, 8, ax=axs[0][0])
        e_min, e_max = (-4, 4)
        if self._bs.is_metal():
            e_min, e_max = (-10, 10)
        for idx, el in enumerate(self._bs.structure.elements, start=1):
            ax = plt.subplot(220 + idx)
            self._make_ticks(ax)
            for b in range(len(data['distances'])):
                for band_idx in range(self._nb_bands):
                    ax.plot(data['distances'][b], data['energy'][str(Spin.up)][b][band_idx], '-', color=[192 / 255, 192 / 255, 192 / 255], linewidth=band_linewidth)
                    if self._bs.is_spin_polarized:
                        ax.plot(data['distances'][b], data['energy'][str(Spin.down)][b][band_idx], '--', color=[128 / 255, 128 / 255, 128 / 255], linewidth=band_linewidth)
                        for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                            markerscale = sum((proj[b][str(Spin.down)][band_idx][j][str(el)][o] for o in proj[b][str(Spin.down)][band_idx][j][str(el)]))
                            ax.plot(data['distances'][b][j], data['energy'][str(Spin.down)][b][band_idx][j], 'bo', markersize=markerscale * 15.0, color=[markerscale, 0.3 * markerscale, 0.4 * markerscale])
                    for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                        markerscale = sum((proj[b][str(Spin.up)][band_idx][j][str(el)][o] for o in proj[b][str(Spin.up)][band_idx][j][str(el)]))
                        ax.plot(data['distances'][b][j], data['energy'][str(Spin.up)][b][band_idx][j], 'o', markersize=markerscale * 15.0, color=[markerscale, 0.3 * markerscale, 0.4 * markerscale])
            if ylim is None:
                if self._bs.is_metal():
                    if zero_to_efermi:
                        ax.set_ylim(e_min, e_max)
                    else:
                        ax.set_ylim(self._bs.efermi + e_min, self._bs.efermi + e_max)
                else:
                    if vbm_cbm_marker:
                        for cbm in data['cbm']:
                            ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                        for vbm in data['vbm']:
                            ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
                    ax.set_ylim(data['vbm'][0][1] + e_min, data['cbm'][0][1] + e_max)
            else:
                ax.set_ylim(ylim)
            ax.set_title(str(el))
        return axs

    def get_elt_projected_plots_color(self, zero_to_efermi=True, elt_ordered=None):
        """Returns a pyplot plot object with one plot where the band structure
        line color depends on the character of the band (along different
        elements). Each element is associated with red, green or blue
        and the corresponding rgb color depending on the character of the band
        is used. The method can only deal with binary and ternary compounds.

        Spin up and spin down are differentiated by a '-' and a '--' line.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            elt_ordered: A list of Element ordered. The first one is red, second green, last blue.

        Raises:
            ValueError: if the number of elements is not 2 or 3.

        Returns:
            a pyplot object
        """
        band_linewidth = 3
        n_elems = len(self._bs.structure.elements)
        if n_elems > 3:
            raise ValueError(f'Can only plot binary and ternary compounds, got {n_elems} elements')
        if elt_ordered is None:
            elt_ordered = self._bs.structure.elements
        proj = self._get_projections_by_branches({e.symbol: ['s', 'p', 'd'] for e in self._bs.structure.elements})
        data = self.bs_plot_data(zero_to_efermi)
        ax = pretty_plot(12, 8)
        spins = [Spin.up]
        if self._bs.is_spin_polarized:
            spins = [Spin.up, Spin.down]
        self._make_ticks(ax)
        for spin in spins:
            for b in range(len(data['distances'])):
                for band_idx in range(self._nb_bands):
                    for j in range(len(data['energy'][str(spin)][b][band_idx]) - 1):
                        sum_e = 0.0
                        for el in elt_ordered:
                            sum_e = sum_e + sum((proj[b][str(spin)][band_idx][j][str(el)][o] for o in proj[b][str(spin)][band_idx][j][str(el)]))
                        if sum_e == 0.0:
                            color = [0.0] * len(elt_ordered)
                        else:
                            color = [sum((proj[b][str(spin)][band_idx][j][str(el)][o] for o in proj[b][str(spin)][band_idx][j][str(el)])) / sum_e for el in elt_ordered]
                        if len(color) == 2:
                            color.append(0.0)
                            color[2] = color[1]
                            color[1] = 0.0
                        sign = '-'
                        if spin == Spin.down:
                            sign = '--'
                        ax.plot([data['distances'][b][j], data['distances'][b][j + 1]], [data['energy'][str(spin)][b][band_idx][j], data['energy'][str(spin)][b][band_idx][j + 1]], sign, color=color, linewidth=band_linewidth)
        if self._bs.is_metal():
            if zero_to_efermi:
                e_min = -10
                e_max = 10
                ax.set_ylim(e_min, e_max)
                ax.set_ylim(self._bs.efermi + e_min, self._bs.efermi + e_max)
        else:
            ax.set_ylim(data['vbm'][0][1] - 4.0, data['cbm'][0][1] + 2.0)
        x_max = data['distances'][-1][-1]
        ax.set_xlim(0, x_max)
        return ax

    def _get_projections_by_branches_patom_pmorb(self, dictio, dictpa, sum_atoms, sum_morbs, selected_branches):
        setos = {'s': 0, 'py': 1, 'pz': 2, 'px': 3, 'dxy': 4, 'dyz': 5, 'dz2': 6, 'dxz': 7, 'dx2': 8, 'f_3': 9, 'f_2': 10, 'f_1': 11, 'f0': 12, 'f1': 13, 'f2': 14, 'f3': 15}
        n_branches = len(self._bs.branches)
        if selected_branches is not None:
            indices = []
            if not isinstance(selected_branches, list):
                raise TypeError("You do not give a correct type of 'selected_branches'. It should be 'list' type.")
            if len(selected_branches) == 0:
                raise ValueError("The 'selected_branches' is empty. We cannot do anything.")
            for index in selected_branches:
                if not isinstance(index, int):
                    raise ValueError("You do not give a correct type of index of symmetry lines. It should be 'int' type")
                if index > n_branches or index < 1:
                    raise ValueError(f'You give a incorrect index of symmetry lines: {index}. The index should be in range of [1, {n_branches}].')
                indices.append(index - 1)
        else:
            indices = range(n_branches)
        proj = self._bs.projections
        proj_br = []
        for index in indices:
            b = self._bs.branches[index]
            print(b)
            if self._bs.is_spin_polarized:
                proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)], str(Spin.down): [[] for _ in range(self._nb_bands)]})
            else:
                proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)]})
            for band_idx in range(self._nb_bands):
                for j in range(b['start_index'], b['end_index'] + 1):
                    edict = {}
                    for elt in dictpa:
                        for anum in dictpa[elt]:
                            edict[f'{elt}{anum}'] = {}
                            for morb in dictio[elt]:
                                edict[f'{elt}{anum}'][morb] = proj[Spin.up][band_idx][j][setos[morb]][anum - 1]
                    proj_br[-1][str(Spin.up)][band_idx].append(edict)
            if self._bs.is_spin_polarized:
                for band_idx in range(self._nb_bands):
                    for j in range(b['start_index'], b['end_index'] + 1):
                        edict = {}
                        for elt in dictpa:
                            for anum in dictpa[elt]:
                                edict[f'{elt}{anum}'] = {}
                                for morb in dictio[elt]:
                                    edict[f'{elt}{anum}'][morb] = proj[Spin.up][band_idx][j][setos[morb]][anum - 1]
                        proj_br[-1][str(Spin.down)][band_idx].append(edict)
        dictio_d, dictpa_d = self._summarize_keys_for_plot(dictio, dictpa, sum_atoms, sum_morbs)
        if sum_atoms is None and sum_morbs is None:
            proj_br_d = copy.deepcopy(proj_br)
        else:
            proj_br_d = []
            branch = -1
            for index in indices:
                branch += 1
                br = self._bs.branches[index]
                if self._bs.is_spin_polarized:
                    proj_br_d.append({str(Spin.up): [[] for _ in range(self._nb_bands)], str(Spin.down): [[] for _ in range(self._nb_bands)]})
                else:
                    proj_br_d.append({str(Spin.up): [[] for _ in range(self._nb_bands)]})
                if sum_atoms is not None and sum_morbs is None:
                    for band_idx in range(self._nb_bands):
                        for j in range(br['end_index'] - br['start_index'] + 1):
                            atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.up)][band_idx][j])
                            edict = {}
                            for elt in dictpa:
                                if elt in sum_atoms:
                                    for anum in dictpa_d[elt][:-1]:
                                        edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                    edict[elt + dictpa_d[elt][-1]] = {}
                                    for morb in dictio[elt]:
                                        sprojection = 0.0
                                        for anum in sum_atoms[elt]:
                                            sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                        edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                else:
                                    for anum in dictpa_d[elt]:
                                        edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                            proj_br_d[-1][str(Spin.up)][band_idx].append(edict)
                    if self._bs.is_spin_polarized:
                        for band_idx in range(self._nb_bands):
                            for j in range(br['end_index'] - br['start_index'] + 1):
                                atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.down)][band_idx][j])
                                edict = {}
                                for elt in dictpa:
                                    if elt in sum_atoms:
                                        for anum in dictpa_d[elt][:-1]:
                                            edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                        edict[elt + dictpa_d[elt][-1]] = {}
                                        for morb in dictio[elt]:
                                            sprojection = 0.0
                                            for anum in sum_atoms[elt]:
                                                sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                            edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                    else:
                                        for anum in dictpa_d[elt]:
                                            edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                proj_br_d[-1][str(Spin.down)][band_idx].append(edict)
                elif sum_atoms is None and sum_morbs is not None:
                    for band_idx in range(self._nb_bands):
                        for j in range(br['end_index'] - br['start_index'] + 1):
                            atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.up)][band_idx][j])
                            edict = {}
                            for elt in dictpa:
                                if elt in sum_morbs:
                                    for anum in dictpa_d[elt]:
                                        edict[elt + anum] = {}
                                        for morb in dictio_d[elt][:-1]:
                                            edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                        sprojection = 0.0
                                        for morb in sum_morbs[elt]:
                                            sprojection += atoms_morbs[elt + anum][morb]
                                        edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                else:
                                    for anum in dictpa_d[elt]:
                                        edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                            proj_br_d[-1][str(Spin.up)][band_idx].append(edict)
                    if self._bs.is_spin_polarized:
                        for band_idx in range(self._nb_bands):
                            for j in range(br['end_index'] - br['start_index'] + 1):
                                atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.down)][band_idx][j])
                                edict = {}
                                for elt in dictpa:
                                    if elt in sum_morbs:
                                        for anum in dictpa_d[elt]:
                                            edict[elt + anum] = {}
                                            for morb in dictio_d[elt][:-1]:
                                                edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                            sprojection = 0.0
                                            for morb in sum_morbs[elt]:
                                                sprojection += atoms_morbs[elt + anum][morb]
                                            edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                    else:
                                        for anum in dictpa_d[elt]:
                                            edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                proj_br_d[-1][str(Spin.down)][band_idx].append(edict)
                else:
                    for band_idx in range(self._nb_bands):
                        for j in range(br['end_index'] - br['start_index'] + 1):
                            atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.up)][band_idx][j])
                            edict = {}
                            for elt in dictpa:
                                if elt in sum_atoms and elt in sum_morbs:
                                    for anum in dictpa_d[elt][:-1]:
                                        edict[elt + anum] = {}
                                        for morb in dictio_d[elt][:-1]:
                                            edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                        sprojection = 0.0
                                        for morb in sum_morbs[elt]:
                                            sprojection += atoms_morbs[elt + anum][morb]
                                        edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                    edict[elt + dictpa_d[elt][-1]] = {}
                                    for morb in dictio_d[elt][:-1]:
                                        sprojection = 0.0
                                        for anum in sum_atoms[elt]:
                                            sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                        edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                    sprojection = 0.0
                                    for anum in sum_atoms[elt]:
                                        for morb in sum_morbs[elt]:
                                            sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                    edict[elt + dictpa_d[elt][-1]][dictio_d[elt][-1]] = sprojection
                                elif elt in sum_atoms and elt not in sum_morbs:
                                    for anum in dictpa_d[elt][:-1]:
                                        edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                    edict[elt + dictpa_d[elt][-1]] = {}
                                    for morb in dictio[elt]:
                                        sprojection = 0.0
                                        for anum in sum_atoms[elt]:
                                            sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                        edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                elif elt not in sum_atoms and elt in sum_morbs:
                                    for anum in dictpa_d[elt]:
                                        edict[elt + anum] = {}
                                        for morb in dictio_d[elt][:-1]:
                                            edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                        sprojection = 0.0
                                        for morb in sum_morbs[elt]:
                                            sprojection += atoms_morbs[elt + anum][morb]
                                        edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                else:
                                    for anum in dictpa_d[elt]:
                                        edict[elt + anum] = {}
                                        for morb in dictio_d[elt]:
                                            edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                            proj_br_d[-1][str(Spin.up)][band_idx].append(edict)
                    if self._bs.is_spin_polarized:
                        for band_idx in range(self._nb_bands):
                            for j in range(br['end_index'] - br['start_index'] + 1):
                                atoms_morbs = copy.deepcopy(proj_br[branch][str(Spin.down)][band_idx][j])
                                edict = {}
                                for elt in dictpa:
                                    if elt in sum_atoms and elt in sum_morbs:
                                        for anum in dictpa_d[elt][:-1]:
                                            edict[elt + anum] = {}
                                            for morb in dictio_d[elt][:-1]:
                                                edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                            sprojection = 0.0
                                            for morb in sum_morbs[elt]:
                                                sprojection += atoms_morbs[elt + anum][morb]
                                            edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                        edict[elt + dictpa_d[elt][-1]] = {}
                                        for morb in dictio_d[elt][:-1]:
                                            sprojection = 0.0
                                            for anum in sum_atoms[elt]:
                                                sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                            edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                        sprojection = 0.0
                                        for anum in sum_atoms[elt]:
                                            for morb in sum_morbs[elt]:
                                                sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                        edict[elt + dictpa_d[elt][-1]][dictio_d[elt][-1]] = sprojection
                                    elif elt in sum_atoms and elt not in sum_morbs:
                                        for anum in dictpa_d[elt][:-1]:
                                            edict[elt + anum] = copy.deepcopy(atoms_morbs[elt + anum])
                                        edict[elt + dictpa_d[elt][-1]] = {}
                                        for morb in dictio[elt]:
                                            sprojection = 0.0
                                            for anum in sum_atoms[elt]:
                                                sprojection += atoms_morbs[f'{elt}{anum}'][morb]
                                            edict[elt + dictpa_d[elt][-1]][morb] = sprojection
                                    elif elt not in sum_atoms and elt in sum_morbs:
                                        for anum in dictpa_d[elt]:
                                            edict[elt + anum] = {}
                                            for morb in dictio_d[elt][:-1]:
                                                edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                            sprojection = 0.0
                                            for morb in sum_morbs[elt]:
                                                sprojection += atoms_morbs[elt + anum][morb]
                                            edict[elt + anum][dictio_d[elt][-1]] = sprojection
                                    else:
                                        for anum in dictpa_d[elt]:
                                            edict[elt + anum] = {}
                                            for morb in dictio_d[elt]:
                                                edict[elt + anum][morb] = atoms_morbs[elt + anum][morb]
                                proj_br_d[-1][str(Spin.down)][band_idx].append(edict)
        return (proj_br_d, dictio_d, dictpa_d, indices)

    def get_projected_plots_dots_patom_pmorb(self, dictio, dictpa, sum_atoms=None, sum_morbs=None, zero_to_efermi=True, ylim=None, vbm_cbm_marker=False, selected_branches=None, w_h_size=(12, 8), num_column=None):
        """Method returns a plot composed of subplots for different atoms and
        orbitals (subshell orbitals such as 's', 'p', 'd' and 'f' defined by
        azimuthal quantum numbers l = 0, 1, 2 and 3, respectively or
        individual orbitals like 'px', 'py' and 'pz' defined by magnetic
        quantum numbers m = -1, 1 and 0, respectively).
        This is an extension of "get_projected_plots_dots" method.

        Args:
            dictio: The elements and the orbitals you need to project on. The
                format is {Element:[Orbitals]}, for instance:
                {'Cu':['dxy','s','px'],'O':['px','py','pz']} will give projections for Cu on
                orbitals dxy, s, px and for O on orbitals px, py, pz. If you want to sum over all
                individual orbitals of subshell orbitals, for example, 'px', 'py' and 'pz' of O,
                just simply set {'Cu':['dxy','s','px'],'O':['p']} and set sum_morbs (see
                explanations below) as {'O':[p],...}. Otherwise, you will get an error.
            dictpa: The elements and their sites (defined by site numbers) you
                need to project on. The format is {Element: [Site numbers]}, for instance:
                {'Cu':[1,5],'O':[3,4]} will give projections for Cu on site-1 and on site-5, O on
                site-3 and on site-4 in the cell. The correct site numbers of atoms are consistent
                with themselves in the structure computed. Normally, the structure should be totally
                similar with POSCAR file, however, sometimes VASP can rotate or translate the cell.
                Thus, it would be safe if using Vasprun class to get the final_structure and as a
                result, correct index numbers of atoms.
            sum_atoms: Sum projection of the similar atoms together (e.g.: Cu
                on site-1 and Cu on site-5). The format is {Element: [Site numbers]}, for instance:
                {'Cu': [1,5], 'O': [3,4]} means summing projections over Cu on site-1 and Cu on
                site-5 and O on site-3 and on site-4. If you do not want to use this functional,
                just turn it off by setting sum_atoms = None.
            sum_morbs: Sum projections of individual orbitals of similar atoms
                together (e.g.: 'dxy' and 'dxz'). The format is {Element: [individual orbitals]},
                for instance: {'Cu': ['dxy', 'dxz'], 'O': ['px', 'py']} means summing projections
                over 'dxy' and 'dxz' of Cu and 'px' and 'py' of O. If you do not want to use this
                functional, just turn it off by setting sum_morbs = None.
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            ylim: The y-axis limit. Defaults to None.
            vbm_cbm_marker: Whether to plot points to indicate valence band maxima and conduction
                band minima positions. Defaults to False.
            selected_branches: The index of symmetry lines you chose for
                plotting. This can be useful when the number of symmetry lines (in KPOINTS file) are
                manny while you only want to show for certain ones. The format is [index of line],
                for instance: [1, 3, 4] means you just need to do projection along lines number 1, 3
                and 4 while neglecting lines number 2 and so on. By default, this is None type and
                all symmetry lines will be plotted.
            w_h_size: This variable help you to control the width and height
                of figure. By default, width = 12 and height = 8 (inches). The width/height ratio is
                kept the same for subfigures and the size of each depends on how many number of
                subfigures are plotted.
            num_column: This variable help you to manage how the subfigures are
                arranged in the figure by setting up the number of columns of subfigures. The value
                should be an int number. For example, num_column = 3 means you want to plot
                subfigures in 3 columns. By default, num_column = None and subfigures are aligned in
                2 columns.

        Returns:
            A pyplot object with different subfigures for different projections.
            The blue and red colors lines are bands
            for spin up and spin down. The green and cyan dots are projections
            for spin up and spin down. The bigger
            the green or cyan dots in the projected band structures, the higher
            character for the corresponding elements
            and orbitals. List of individual orbitals and their numbers (set up
            by VASP and no special meaning):
            s = 0; py = 1 pz = 2 px = 3; dxy = 4 dyz = 5 dz2 = 6 dxz = 7 dx2 = 8;
            f_3 = 9 f_2 = 10 f_1 = 11 f0 = 12 f1 = 13 f2 = 14 f3 = 15
        """
        dictio, sum_morbs = self._Orbitals_SumOrbitals(dictio, sum_morbs)
        dictpa, sum_atoms, n_figs = self._number_of_subfigures(dictio, dictpa, sum_atoms, sum_morbs)
        print(f'Number of subfigures: {n_figs}')
        if n_figs > 9:
            print(f'The number of subfigures {n_figs} might be too manny and the implementation might take a long time.\n A smaller number or a plot with selected symmetry lines (selected_branches) might be better.')
        band_linewidth = 0.5
        ax = pretty_plot(w_h_size[0], w_h_size[1])
        proj_br_d, dictio_d, dictpa_d, branches = self._get_projections_by_branches_patom_pmorb(dictio, dictpa, sum_atoms, sum_morbs, selected_branches)
        data = self.bs_plot_data(zero_to_efermi)
        e_min = -4
        e_max = 4
        if self._bs.is_metal():
            e_min = -10
            e_max = 10
        count = 0
        for elt in dictpa_d:
            for numa in dictpa_d[elt]:
                for o in dictio_d[elt]:
                    count += 1
                    if num_column is None:
                        if n_figs == 1:
                            plt.subplot(1, 1, 1)
                        else:
                            row = n_figs // 2
                            if n_figs % 2 == 0:
                                plt.subplot(row, 2, count)
                            else:
                                plt.subplot(row + 1, 2, count)
                    elif isinstance(num_column, int):
                        row = n_figs / num_column
                        if n_figs % num_column == 0:
                            plt.subplot(row, num_column, count)
                        else:
                            plt.subplot(row + 1, num_column, count)
                    else:
                        raise ValueError("The invalid 'num_column' is assigned. It should be an integer.")
                    ax, shift = self._make_ticks_selected(ax, branches)
                    br = -1
                    for b in branches:
                        br += 1
                        for band_idx in range(self._nb_bands):
                            ax.plot([x - shift[br] for x in data['distances'][b]], [data['energy'][str(Spin.up)][b][band_idx][j] for j in range(len(data['distances'][b]))], 'b-', linewidth=band_linewidth)
                            if self._bs.is_spin_polarized:
                                ax.plot([x - shift[br] for x in data['distances'][b]], [data['energy'][str(Spin.down)][b][band_idx][j] for j in range(len(data['distances'][b]))], 'r--', linewidth=band_linewidth)
                                for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                                    ax.plot(data['distances'][b][j] - shift[br], data['energy'][str(Spin.down)][b][band_idx][j], 'co', markersize=proj_br_d[br][str(Spin.down)][band_idx][j][elt + numa][o] * 15.0)
                            for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                                ax.plot(data['distances'][b][j] - shift[br], data['energy'][str(Spin.up)][b][band_idx][j], 'go', markersize=proj_br_d[br][str(Spin.up)][band_idx][j][elt + numa][o] * 15.0)
                    if ylim is None:
                        if self._bs.is_metal():
                            if zero_to_efermi:
                                ax.set_ylim(e_min, e_max)
                            else:
                                ax.set_ylim(self._bs.efermi + e_min, self._bs._efermi + e_max)
                        else:
                            if vbm_cbm_marker:
                                for cbm in data['cbm']:
                                    ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                                for vbm in data['vbm']:
                                    ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
                            ax.set_ylim(data['vbm'][0][1] + e_min, data['cbm'][0][1] + e_max)
                    else:
                        ax.set_ylim(ylim)
                    ax.set_title(f'{elt} {numa} {o}')
        return ax

    @classmethod
    def _Orbitals_SumOrbitals(cls, dictio, sum_morbs):
        all_orbitals = ['s', 'p', 'd', 'f', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2', 'dz2', 'f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']
        individual_orbs = {'p': ['px', 'py', 'pz'], 'd': ['dxy', 'dyz', 'dxz', 'dx2', 'dz2'], 'f': ['f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']}
        if not isinstance(dictio, dict):
            raise TypeError("The invalid type of 'dictio' was bound. It should be dict type.")
        if len(dictio) == 0:
            raise KeyError("The 'dictio' is empty. We cannot do anything.")
        for elt in dictio:
            if Element.is_valid_symbol(elt):
                if isinstance(dictio[elt], list):
                    if len(dictio[elt]) == 0:
                        raise ValueError(f'The dictio[{elt}] is empty. We cannot do anything')
                    for orb in dictio[elt]:
                        if not isinstance(orb, str):
                            raise ValueError(f"The invalid format of orbitals is in 'dictio[{elt}]': {orb}. They should be string.")
                        if orb not in all_orbitals:
                            raise ValueError(f"The invalid name of orbital is given in 'dictio[{elt}]'.")
                        if orb in individual_orbs and len(set(dictio[elt]).intersection(individual_orbs[orb])) != 0:
                            raise ValueError(f"The 'dictio[{elt}]' contains orbitals repeated.")
                    nelems = Counter(dictio[elt]).values()
                    if sum(nelems) > len(nelems):
                        raise ValueError(f'You put in at least two similar orbitals in dictio[{elt}].')
                else:
                    raise TypeError(f"The invalid type of value was put into 'dictio[{elt}]'. It should be list type.")
            else:
                raise KeyError(f"The invalid element was put into 'dictio' as a key: {elt}")
        if sum_morbs is None:
            print('You do not want to sum projection over orbitals.')
        elif not isinstance(sum_morbs, dict):
            raise TypeError("The invalid type of 'sum_orbs' was bound. It should be dict or 'None' type.")
        elif len(sum_morbs) == 0:
            raise KeyError("The 'sum_morbs' is empty. We cannot do anything")
        else:
            for elt in sum_morbs:
                if Element.is_valid_symbol(elt):
                    if isinstance(sum_morbs[elt], list):
                        for orb in sum_morbs[elt]:
                            if not isinstance(orb, str):
                                raise TypeError(f"The invalid format of orbitals is in 'sum_morbs[{elt}]': {orb}. They should be string.")
                            if orb not in all_orbitals:
                                raise ValueError(f"The invalid name of orbital in 'sum_morbs[{elt}]' is given.")
                            if orb in individual_orbs and len(set(sum_morbs[elt]) & set(individual_orbs[orb])) != 0:
                                raise ValueError(f"The 'sum_morbs[{elt}]' contains orbitals repeated.")
                        nelems = Counter(sum_morbs[elt]).values()
                        if sum(nelems) > len(nelems):
                            raise ValueError(f'You put in at least two similar orbitals in sum_morbs[{elt}].')
                    else:
                        raise TypeError(f"The invalid type of value was put into 'sum_morbs[{elt}]'. It should be list type.")
                    if elt not in dictio:
                        raise ValueError(f"You cannot sum projection over orbitals of atoms {elt!r} because they are not mentioned in 'dictio'.")
                else:
                    raise KeyError(f"The invalid element was put into 'sum_morbs' as a key: {elt}")
        for elt in dictio:
            if len(dictio[elt]) == 1:
                if len(dictio[elt][0]) > 1:
                    if elt in sum_morbs:
                        raise ValueError(f'You cannot sum projection over one individual orbital {dictio[elt][0]!r} of {elt!r}.')
                elif sum_morbs is None:
                    pass
                elif elt not in sum_morbs:
                    print(f'You do not want to sum projection over orbitals of element: {elt}')
                else:
                    if len(sum_morbs[elt]) == 0:
                        raise ValueError(f'The empty list is an invalid value for sum_morbs[{elt}].')
                    if len(sum_morbs[elt]) > 1:
                        for orb in sum_morbs[elt]:
                            if dictio[elt][0] not in orb:
                                raise ValueError(f"The invalid orbital {orb!r} was put into 'sum_morbs[{elt}]'.")
                    else:
                        if orb == 's' or len(orb) > 1:
                            raise ValueError(f'The invalid orbital {orb!r} was put into sum_orbs[{elt!r}].')
                        sum_morbs[elt] = individual_orbs[dictio[elt][0]]
                        dictio[elt] = individual_orbs[dictio[elt][0]]
            else:
                duplicate = copy.deepcopy(dictio[elt])
                for orb in dictio[elt]:
                    if orb in individual_orbs:
                        duplicate.remove(orb)
                        duplicate += individual_orbs[orb]
                dictio[elt] = copy.deepcopy(duplicate)
                if sum_morbs is None:
                    pass
                elif elt not in sum_morbs:
                    print(f'You do not want to sum projection over orbitals of element: {elt}')
                else:
                    if len(sum_morbs[elt]) == 0:
                        raise ValueError(f'The empty list is an invalid value for sum_morbs[{elt}].')
                    if len(sum_morbs[elt]) == 1:
                        orb = sum_morbs[elt][0]
                        if orb == 's':
                            raise ValueError("We do not sum projection over only 's' orbital of the same type of element.")
                        if orb in individual_orbs:
                            sum_morbs[elt].pop(0)
                            sum_morbs[elt] += individual_orbs[orb]
                        else:
                            raise ValueError(f'You never sum projection over one orbital in sum_morbs[{elt}]')
                    else:
                        duplicate = copy.deepcopy(sum_morbs[elt])
                        for orb in sum_morbs[elt]:
                            if orb in individual_orbs:
                                duplicate.remove(orb)
                                duplicate += individual_orbs[orb]
                        sum_morbs[elt] = copy.deepcopy(duplicate)
                    for orb in sum_morbs[elt]:
                        if orb not in dictio[elt]:
                            raise ValueError(f'The orbitals of sum_morbs[{elt}] conflict with those of dictio[{elt}].')
        return (dictio, sum_morbs)

    def _number_of_subfigures(self, dictio, dictpa, sum_atoms, sum_morbs):
        if not isinstance(dictpa, dict):
            raise TypeError("The invalid type of 'dictpa' was bound. It should be dict type.")
        if len(dictpa) == 0:
            raise KeyError("The 'dictpa' is empty. We cannot do anything.")
        for elt in dictpa:
            if Element.is_valid_symbol(elt):
                if isinstance(dictpa[elt], list):
                    if len(dictpa[elt]) == 0:
                        raise ValueError(f'The dictpa[{elt}] is empty. We cannot do anything')
                    _sites = self._bs.structure.sites
                    indices = []
                    for site_idx in range(len(_sites)):
                        if next(iter(_sites[site_idx]._species)) == Element(elt):
                            indices.append(site_idx + 1)
                    for number in dictpa[elt]:
                        if isinstance(number, str):
                            if number.lower() == 'all':
                                dictpa[elt] = indices
                                print(f'You want to consider all {elt!r} atoms.')
                                break
                            raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                        if isinstance(number, int):
                            if number not in indices:
                                raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                        else:
                            raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                    nelems = Counter(dictpa[elt]).values()
                    if sum(nelems) > len(nelems):
                        raise ValueError(f"You put at least two similar site numbers into 'dictpa[{elt}]'.")
                else:
                    raise TypeError(f"The invalid type of value was put into 'dictpa[{elt}]'. It should be list type.")
            else:
                raise KeyError(f"The invalid element was put into 'dictpa' as a key: {elt}")
        if len(list(dictio)) != len(list(dictpa)):
            raise KeyError("The number of keys in 'dictio' and 'dictpa' are not the same.")
        for elt in dictio:
            if elt not in dictpa:
                raise KeyError(f'The element {elt!r} is not in both dictpa and dictio.')
        for elt in dictpa:
            if elt not in dictio:
                raise KeyError(f'The element {elt!r} in not in both dictpa and dictio.')
        if sum_atoms is None:
            print('You do not want to sum projection over atoms.')
        elif not isinstance(sum_atoms, dict):
            raise TypeError("The invalid type of 'sum_atoms' was bound. It should be dict type.")
        elif len(sum_atoms) == 0:
            raise KeyError("The 'sum_atoms' is empty. We cannot do anything.")
        else:
            for elt in sum_atoms:
                if Element.is_valid_symbol(elt):
                    if isinstance(sum_atoms[elt], list):
                        if len(sum_atoms[elt]) == 0:
                            raise ValueError(f'The sum_atoms[{elt}] is empty. We cannot do anything')
                        _sites = self._bs.structure.sites
                        indices = []
                        for site_idx in range(len(_sites)):
                            if next(iter(_sites[site_idx]._species)) == Element(elt):
                                indices.append(site_idx + 1)
                        for number in sum_atoms[elt]:
                            if isinstance(number, str):
                                if number.lower() == 'all':
                                    sum_atoms[elt] = indices
                                    print(f'You want to sum projection over all {elt!r} atoms.')
                                    break
                                raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                            if isinstance(number, int):
                                if number not in indices:
                                    raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                                if number not in dictpa[elt]:
                                    raise ValueError(f'You cannot sum projection with atom number {number!r} because it is not mentioned in dicpta[{elt}]')
                            else:
                                raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                        nelems = Counter(sum_atoms[elt]).values()
                        if sum(nelems) > len(nelems):
                            raise ValueError(f"You put at least two similar site numbers into 'sum_atoms[{elt}]'.")
                    else:
                        raise TypeError(f"The invalid type of value was put into 'sum_atoms[{elt}]'. It should be list type.")
                    if elt not in dictpa:
                        raise ValueError(f"You cannot sum projection over atoms {elt!r} because it is not mentioned in 'dictio'.")
                else:
                    raise KeyError(f"The invalid element was put into 'sum_atoms' as a key: {elt}")
                if len(sum_atoms[elt]) == 1:
                    raise ValueError(f'We do not sum projection over only one atom: {elt}')
        max_number_figs = 0
        decrease = 0
        for elt in dictio:
            max_number_figs += len(dictio[elt]) * len(dictpa[elt])
        if sum_atoms is None and sum_morbs is None:
            number_figs = max_number_figs
        elif sum_atoms is not None and sum_morbs is None:
            for elt in sum_atoms:
                decrease += (len(sum_atoms[elt]) - 1) * len(dictio[elt])
            number_figs = max_number_figs - decrease
        elif sum_atoms is None and sum_morbs is not None:
            for elt in sum_morbs:
                decrease += (len(sum_morbs[elt]) - 1) * len(dictpa[elt])
            number_figs = max_number_figs - decrease
        elif sum_atoms is not None and sum_morbs is not None:
            for elt in sum_atoms:
                decrease += (len(sum_atoms[elt]) - 1) * len(dictio[elt])
            for elt in sum_morbs:
                if elt in sum_atoms:
                    decrease += (len(sum_morbs[elt]) - 1) * (len(dictpa[elt]) - len(sum_atoms[elt]) + 1)
                else:
                    decrease += (len(sum_morbs[elt]) - 1) * len(dictpa[elt])
            number_figs = max_number_figs - decrease
        else:
            raise ValueError("Invalid format of 'sum_atoms' and 'sum_morbs'.")
        return (dictpa, sum_atoms, number_figs)

    def _summarize_keys_for_plot(self, dictio, dictpa, sum_atoms, sum_morbs):
        individual_orbs = {'p': ['px', 'py', 'pz'], 'd': ['dxy', 'dyz', 'dxz', 'dx2', 'dz2'], 'f': ['f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']}

        def number_label(list_numbers):
            list_numbers = sorted(list_numbers)
            divide = [[]]
            divide[0].append(list_numbers[0])
            group = 0
            for idx in range(1, len(list_numbers)):
                if list_numbers[idx] == list_numbers[idx - 1] + 1:
                    divide[group].append(list_numbers[idx])
                else:
                    group += 1
                    divide.append([list_numbers[idx]])
            label = ''
            for elem in divide:
                if len(elem) > 1:
                    label += f'{elem[0]}-{elem[-1]},'
                else:
                    label += f'{elem[0]},'
            return label[:-1]

        def orbital_label(list_orbitals):
            divide = {}
            for orb in list_orbitals:
                if orb[0] in divide:
                    divide[orb[0]].append(orb)
                else:
                    divide[orb[0]] = []
                    divide[orb[0]].append(orb)
            label = ''
            for elem, orbs in divide.items():
                if elem == 's':
                    label += 's,'
                elif len(orbs) == len(individual_orbs[elem]):
                    label += elem + ','
                else:
                    orb_label = [orb[1:] for orb in orbs]
                    label += f'{elem}{str(orb_label).replace('[', '').replace(']', '').replace(', ', '-')},'
            return label[:-1]
        if sum_atoms is None and sum_morbs is None:
            dictio_d = dictio
            dictpa_d = {elt: [str(anum) for anum in dictpa[elt]] for elt in dictpa}
        elif sum_atoms is not None and sum_morbs is None:
            dictio_d = dictio
            dictpa_d = {}
            for elt in dictpa:
                dictpa_d[elt] = []
                if elt in sum_atoms:
                    _sites = self._bs.structure.sites
                    indices = []
                    for site_idx in range(len(_sites)):
                        if next(iter(_sites[site_idx]._species)) == Element(elt):
                            indices.append(site_idx + 1)
                    flag_1 = len(set(dictpa[elt]).intersection(indices))
                    flag_2 = len(set(sum_atoms[elt]).intersection(indices))
                    if flag_1 == len(indices) and flag_2 == len(indices):
                        dictpa_d[elt].append('all')
                    else:
                        for anum in dictpa[elt]:
                            if anum not in sum_atoms[elt]:
                                dictpa_d[elt].append(str(anum))
                        label = number_label(sum_atoms[elt])
                        dictpa_d[elt].append(label)
                else:
                    for anum in dictpa[elt]:
                        dictpa_d[elt].append(str(anum))
        elif sum_atoms is None and sum_morbs is not None:
            dictio_d = {}
            for elt in dictio:
                dictio_d[elt] = []
                if elt in sum_morbs:
                    for morb in dictio[elt]:
                        if morb not in sum_morbs[elt]:
                            dictio_d[elt].append(morb)
                    label = orbital_label(sum_morbs[elt])
                    dictio_d[elt].append(label)
                else:
                    dictio_d[elt] = dictio[elt]
            dictpa_d = {elt: [str(anum) for anum in dictpa[elt]] for elt in dictpa}
        else:
            dictio_d = {}
            for elt in dictio:
                dictio_d[elt] = []
                if elt in sum_morbs:
                    for morb in dictio[elt]:
                        if morb not in sum_morbs[elt]:
                            dictio_d[elt].append(morb)
                    label = orbital_label(sum_morbs[elt])
                    dictio_d[elt].append(label)
                else:
                    dictio_d[elt] = dictio[elt]
            dictpa_d = {}
            for elt in dictpa:
                dictpa_d[elt] = []
                if elt in sum_atoms:
                    _sites = self._bs.structure.sites
                    indices = []
                    for site_idx in range(len(_sites)):
                        if next(iter(_sites[site_idx]._species)) == Element(elt):
                            indices.append(site_idx + 1)
                    flag_1 = len(set(dictpa[elt]).intersection(indices))
                    flag_2 = len(set(sum_atoms[elt]).intersection(indices))
                    if flag_1 == len(indices) and flag_2 == len(indices):
                        dictpa_d[elt].append('all')
                    else:
                        for anum in dictpa[elt]:
                            if anum not in sum_atoms[elt]:
                                dictpa_d[elt].append(str(anum))
                        label = number_label(sum_atoms[elt])
                        dictpa_d[elt].append(label)
                else:
                    for anum in dictpa[elt]:
                        dictpa_d[elt].append(str(anum))
        return (dictio_d, dictpa_d)

    def _make_ticks_selected(self, ax: plt.Axes, branches: list[int]) -> tuple[plt.Axes, list[float]]:
        """Utility private method to add ticks to a band structure with selected branches."""
        if not ax.figure:
            fig = plt.figure()
            ax.set_figure(fig)
        ticks = self.get_ticks()
        distance = []
        label = []
        rm_elems = []
        for idx in range(1, len(ticks['distance'])):
            if ticks['label'][idx] == ticks['label'][idx - 1]:
                rm_elems.append(idx)
        for idx in range(len(ticks['distance'])):
            if idx not in rm_elems:
                distance.append(ticks['distance'][idx])
                label.append(ticks['label'][idx])
        l_branches = [distance[i] - distance[i - 1] for i in range(1, len(distance))]
        n_distance = []
        n_label = []
        for branch in branches:
            n_distance.append(l_branches[branch])
            if '$\\mid$' not in label[branch] and '$\\mid$' not in label[branch + 1]:
                n_label.append([label[branch], label[branch + 1]])
            elif '$\\mid$' in label[branch] and '$\\mid$' not in label[branch + 1]:
                n_label.append([label[branch].split('$')[-1], label[branch + 1]])
            elif '$\\mid$' not in label[branch] and '$\\mid$' in label[branch + 1]:
                n_label.append([label[branch], label[branch + 1].split('$')[0]])
            else:
                n_label.append([label[branch].split('$')[-1], label[branch + 1].split('$')[0]])
        f_distance: list[float] = []
        rf_distance: list[float] = []
        f_label: list[str] = []
        f_label.extend((n_label[0][0], n_label[0][1]))
        f_distance.extend((0.0, n_distance[0]))
        rf_distance.extend((0.0, n_distance[0]))
        length = n_distance[0]
        for idx in range(1, len(n_distance)):
            if n_label[idx][0] == n_label[idx - 1][1]:
                f_distance.extend((length, length + n_distance[idx]))
                f_label.extend((n_label[idx][0], n_label[idx][1]))
            else:
                f_distance.append(length + n_distance[idx])
                f_label[-1] = n_label[idx - 1][1] + '$\\mid$' + n_label[idx][0]
                f_label.append(n_label[idx][1])
            rf_distance.append(length + n_distance[idx])
            length += n_distance[idx]
        uniq_d = []
        uniq_l = []
        temp_ticks = list(zip(f_distance, f_label))
        for idx, tick in enumerate(temp_ticks):
            if idx == 0:
                uniq_d.append(tick[0])
                uniq_l.append(tick[1])
                logger.debug(f'Adding label {tick[0]} at {tick[1]}')
            elif tick[1] == temp_ticks[idx - 1][1]:
                logger.debug(f'Skipping label {tick[1]}')
            else:
                logger.debug(f'Adding label {tick[0]} at {tick[1]}')
                uniq_d.append(tick[0])
                uniq_l.append(tick[1])
        logger.debug(f'Unique labels are {list(zip(uniq_d, uniq_l))}')
        ax.set_xticks(uniq_d)
        ax.set_xticklabels(uniq_l)
        for idx in range(len(f_label)):
            if f_label[idx] is not None:
                if idx != 0:
                    if f_label[idx] == f_label[idx - 1]:
                        logger.debug(f'already print label... skipping label {f_label[idx]}')
                    else:
                        logger.debug(f'Adding a line at {f_distance[idx]} for label {f_label[idx]}')
                        ax.axvline(f_distance[idx], color='k')
                else:
                    logger.debug(f'Adding a line at {f_distance[idx]} for label {f_label[idx]}')
                    ax.axvline(f_distance[idx], color='k')
        shift = []
        br = -1
        for branch in branches:
            br += 1
            shift.append(distance[branch] - rf_distance[br])
        return (ax, shift)