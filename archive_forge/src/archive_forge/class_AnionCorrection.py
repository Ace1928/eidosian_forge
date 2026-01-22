from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
@cached_class
class AnionCorrection(Correction):
    """Correct anion energies to obtain the right formation energies. Note that
    this depends on calculations being run within the same input set.

    Used by legacy MaterialsProjectCompatibility and MITCompatibility.
    """

    def __init__(self, config_file, correct_peroxide=True):
        """
        Args:
            config_file: Path to the selected compatibility.yaml config file.
            correct_peroxide: Specify whether peroxide/superoxide/ozonide
                corrections are to be applied or not.
        """
        config = loadfn(config_file)
        self.oxide_correction = config['OxideCorrections']
        self.sulfide_correction = config.get('SulfideCorrections', defaultdict(float))
        self.name = config['Name']
        self.correct_peroxide = correct_peroxide

    def get_correction(self, entry) -> ufloat:
        """
        Args:
            entry: A ComputedEntry/ComputedStructureEntry.

        Returns:
            Correction.
        """
        comp = entry.composition
        if len(comp) == 1:
            return ufloat(0.0, 0.0)
        correction = ufloat(0.0, 0.0)
        if Element('S') in comp:
            sf_type = 'sulfide'
            if entry.data.get('sulfide_type'):
                sf_type = entry.data['sulfide_type']
            elif hasattr(entry, 'structure'):
                warnings.warn(sf_type)
                sf_type = sulfide_type(entry.structure)
            if sf_type == 'polysulfide':
                sf_type = 'sulfide'
            if sf_type in self.sulfide_correction:
                correction += self.sulfide_correction[sf_type] * comp['S']
        if Element('O') in comp:
            if self.correct_peroxide:
                if entry.data.get('oxide_type'):
                    if entry.data['oxide_type'] in self.oxide_correction:
                        ox_corr = self.oxide_correction[entry.data['oxide_type']]
                        correction += ox_corr * comp['O']
                    if entry.data['oxide_type'] == 'hydroxide':
                        ox_corr = self.oxide_correction['oxide']
                        correction += ox_corr * comp['O']
                elif hasattr(entry, 'structure'):
                    ox_type, n_bonds = oxide_type(entry.structure, 1.05, return_nbonds=True)
                    if ox_type in self.oxide_correction:
                        correction += self.oxide_correction[ox_type] * n_bonds
                    elif ox_type == 'hydroxide':
                        correction += self.oxide_correction['oxide'] * comp['O']
                else:
                    warnings.warn('No structure or oxide_type parameter present. Note that peroxide/superoxide corrections are not as reliable and relies only on detection of special formulas, e.g., Li2O2.')
                    rform = entry.reduced_formula
                    if rform in UCorrection.common_peroxides:
                        correction += self.oxide_correction['peroxide'] * comp['O']
                    elif rform in UCorrection.common_superoxides:
                        correction += self.oxide_correction['superoxide'] * comp['O']
                    elif rform in UCorrection.ozonides:
                        correction += self.oxide_correction['ozonide'] * comp['O']
                    elif Element('O') in comp.elements and len(comp.elements) > 1:
                        correction += self.oxide_correction['oxide'] * comp['O']
            else:
                correction += self.oxide_correction['oxide'] * comp['O']
        return correction

    def __str__(self):
        return f'{self.name} Anion Correction'