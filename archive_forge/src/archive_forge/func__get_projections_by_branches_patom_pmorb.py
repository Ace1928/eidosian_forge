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