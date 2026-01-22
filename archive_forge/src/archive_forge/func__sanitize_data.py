from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
def _sanitize_data(self, data):
    """
        Some CIF files do not conform to spec. This function corrects
        known issues, particular in regards to Springer materials/
        Pauling files.

        This function is here so that CifParser can assume its
        input conforms to spec, simplifying its implementation.

        Args:
            data: CifBlock

        Returns:
            data CifBlock
        """
    if '_atom_site_attached_hydrogens' in data.data:
        attached_hydrogens = [str2float(x) for x in data.data['_atom_site_attached_hydrogens'] if str2float(x) != 0]
        if len(attached_hydrogens) > 0:
            self.warnings.append('Structure has implicit hydrogens defined, parsed structure unlikely to be suitable for use in calculations unless hydrogens added.')
    if '_atom_site_type_symbol' in data.data:
        idxs_to_remove = []
        new_atom_site_label = []
        new_atom_site_type_symbol = []
        new_atom_site_occupancy = []
        new_fract_x = []
        new_fract_y = []
        new_fract_z = []
        for idx, el_row in enumerate(data['_atom_site_label']):
            if len(data['_atom_site_type_symbol'][idx].split(' + ')) > len(el_row.split(' + ')):
                els_occu = {}
                symbol_str = data['_atom_site_type_symbol'][idx]
                symbol_str_lst = symbol_str.split(' + ')
                for elocc_idx, sym in enumerate(symbol_str_lst):
                    symbol_str_lst[elocc_idx] = re.sub('\\([0-9]*\\)', '', sym.strip())
                    els_occu[str(re.findall('\\D+', symbol_str_lst[elocc_idx].strip())[1]).replace('<sup>', '')] = float('0' + re.findall('\\.?\\d+', symbol_str_lst[elocc_idx].strip())[1])
                x = str2float(data['_atom_site_fract_x'][idx])
                y = str2float(data['_atom_site_fract_y'][idx])
                z = str2float(data['_atom_site_fract_z'][idx])
                for et, occu in els_occu.items():
                    new_atom_site_label.append(f'{et}_fix{len(new_atom_site_label)}')
                    new_atom_site_type_symbol.append(et)
                    new_atom_site_occupancy.append(str(occu))
                    new_fract_x.append(str(x))
                    new_fract_y.append(str(y))
                    new_fract_z.append(str(z))
                idxs_to_remove.append(idx)
        for original_key in data.data:
            if isinstance(data.data[original_key], list):
                for idx in sorted(idxs_to_remove, reverse=True):
                    del data.data[original_key][idx]
        if len(idxs_to_remove) > 0:
            self.warnings.append('Pauling file corrections applied.')
            data.data['_atom_site_label'] += new_atom_site_label
            data.data['_atom_site_type_symbol'] += new_atom_site_type_symbol
            data.data['_atom_site_occupancy'] += new_atom_site_occupancy
            data.data['_atom_site_fract_x'] += new_fract_x
            data.data['_atom_site_fract_y'] += new_fract_y
            data.data['_atom_site_fract_z'] += new_fract_z
    if self.feature_flags['magcif']:
        correct_keys = ['_space_group_symop_magn_operation.xyz', '_space_group_symop_magn_centering.xyz', '_space_group_magn.name_BNS', '_space_group_magn.number_BNS', '_atom_site_moment_crystalaxis_x', '_atom_site_moment_crystalaxis_y', '_atom_site_moment_crystalaxis_z', '_atom_site_moment_label']
        changes_to_make = {}
        for original_key in data.data:
            for correct_key in correct_keys:
                trial_key = '_'.join(correct_key.split('.'))
                test_key = '_'.join(original_key.split('.'))
                if trial_key == test_key:
                    changes_to_make[correct_key] = original_key
        for correct_key, original_key in changes_to_make.items():
            data.data[correct_key] = data.data[original_key]
        renamed_keys = {'_magnetic_space_group.transform_to_standard_Pp_abc': '_space_group_magn.transform_BNS_Pp_abc'}
        changes_to_make = {}
        for interim_key, final_key in renamed_keys.items():
            if data.data.get(interim_key):
                changes_to_make[final_key] = interim_key
        if len(changes_to_make) > 0:
            self.warnings.append('Keys changed to match new magCIF specification.')
        for final_key, interim_key in changes_to_make.items():
            data.data[final_key] = data.data[interim_key]
    important_fracs = (1 / 3, 2 / 3)
    fracs_to_change = {}
    for label in ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'):
        if label in data.data:
            for idx, frac in enumerate(data.data[label]):
                try:
                    frac = str2float(frac)
                except Exception:
                    continue
                for comparison_frac in important_fracs:
                    if abs(1 - frac / comparison_frac) < self._frac_tolerance:
                        fracs_to_change[label, idx] = str(comparison_frac)
    if fracs_to_change:
        self.warnings.append(f'{len(fracs_to_change)} fractional coordinates rounded to ideal values to avoid issues with finite precision.')
        for (label, idx), val in fracs_to_change.items():
            data.data[label][idx] = val
    return data