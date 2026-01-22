import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
class CastepCell(CastepInputFile):
    """CastepCell abstracts all setting that go into the .cell file"""
    _keyword_conflicts = [{'kpoint_mp_grid', 'kpoint_mp_spacing', 'kpoint_list', 'kpoints_mp_grid', 'kpoints_mp_spacing', 'kpoints_list'}, {'bs_kpoint_mp_grid', 'bs_kpoint_mp_spacing', 'bs_kpoint_list', 'bs_kpoint_path', 'bs_kpoints_mp_grid', 'bs_kpoints_mp_spacing', 'bs_kpoints_list', 'bs_kpoints_path'}, {'spectral_kpoint_mp_grid', 'spectral_kpoint_mp_spacing', 'spectral_kpoint_list', 'spectral_kpoint_path', 'spectral_kpoints_mp_grid', 'spectral_kpoints_mp_spacing', 'spectral_kpoints_list', 'spectral_kpoints_path'}, {'phonon_kpoint_mp_grid', 'phonon_kpoint_mp_spacing', 'phonon_kpoint_list', 'phonon_kpoint_path', 'phonon_kpoints_mp_grid', 'phonon_kpoints_mp_spacing', 'phonon_kpoints_list', 'phonon_kpoints_path'}, {'fine_phonon_kpoint_mp_grid', 'fine_phonon_kpoint_mp_spacing', 'fine_phonon_kpoint_list', 'fine_phonon_kpoint_path'}, {'magres_kpoint_mp_grid', 'magres_kpoint_mp_spacing', 'magres_kpoint_list', 'magres_kpoint_path'}, {'elnes_kpoint_mp_grid', 'elnes_kpoint_mp_spacing', 'elnes_kpoint_list', 'elnes_kpoint_path'}, {'optics_kpoint_mp_grid', 'optics_kpoint_mp_spacing', 'optics_kpoint_list', 'optics_kpoint_path'}, {'supercell_kpoint_mp_grid', 'supercell_kpoint_mp_spacing', 'supercell_kpoint_list', 'supercell_kpoint_path'}]

    def __init__(self, castep_keywords, keyword_tolerance=1):
        self._castep_version = castep_keywords.castep_version
        CastepInputFile.__init__(self, castep_keywords.CastepCellDict(), keyword_tolerance)

    @property
    def castep_version(self):
        return self._castep_version

    def _parse_species_pot(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            value = [value]
        if hasattr(value, '__getitem__'):
            pspots = [tuple(map(str.strip, x)) for x in value]
            if not all(map(lambda x: len(x) == 2, value)):
                warnings.warn('Please specify pseudopotentials in python as a tuple or a list of tuples formatted like: (species, file), e.g. ("O", "path-to/O_OTFG.usp") Anything else will be ignored')
                return None
        text_block = self._options['species_pot'].value
        text_block = text_block if text_block else ''
        for pp in pspots:
            text_block = re.sub('\\n?\\s*%s\\s+.*' % pp[0], '', text_block)
            if pp[1]:
                text_block += '\n%s %s' % pp
        return text_block

    def _parse_symmetry_ops(self, value):
        if not isinstance(value, tuple) or not len(value) == 2 or (not value[0].shape[1:] == (3, 3)) or (not value[1].shape[1:] == (3,)) or (not value[0].shape[0] == value[1].shape[0]):
            warnings.warn('Invalid symmetry_ops block, skipping')
            return
        text_block = ''
        for op_i, (op_rot, op_tranls) in enumerate(zip(*value)):
            text_block += '\n'.join([' '.join([str(x) for x in row]) for row in op_rot])
            text_block += '\n'
            text_block += ' '.join([str(x) for x in op_tranls])
            text_block += '\n\n'
        return text_block

    def _parse_positions_abs_intermediate(self, value):
        return _parse_tss_block(value)

    def _parse_positions_abs_product(self, value):
        return _parse_tss_block(value)

    def _parse_positions_frac_intermediate(self, value):
        return _parse_tss_block(value, True)

    def _parse_positions_frac_product(self, value):
        return _parse_tss_block(value, True)