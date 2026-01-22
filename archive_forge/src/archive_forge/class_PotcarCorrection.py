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
class PotcarCorrection(Correction):
    """Checks that POTCARs are valid within a pre-defined input set. This
    ensures that calculations performed using different InputSets are not
    compared against each other.

    Entry.parameters must contain a "potcar_symbols" key that is a list of
    all POTCARs used in the run. Again, using the example of an Fe2O3 run
    using Materials Project parameters, this would look like
    entry.parameters["potcar_symbols"] = ['PAW_PBE Fe_pv 06Sep2000',
    'PAW_PBE O 08Apr2002'].
    """

    def __init__(self, input_set: type[VaspInputSet], check_potcar: bool=True, check_hash: bool=False) -> None:
        """
        Args:
            input_set (InputSet): object used to generate the runs (used to check
                for correct potcar symbols).
            check_potcar (bool): If False, bypass the POTCAR check altogether. Defaults to True.
                Can also be disabled globally by running `pmg config --add PMG_POTCAR_CHECKS false`.
            check_hash (bool): If True, uses the potcar hash to check for valid
                potcars. If false, uses the potcar symbol (less reliable). Defaults to False.

        Raises:
            ValueError: if check_potcar=True and entry does not contain "potcar_symbols" key.
        """
        potcar_settings = input_set.CONFIG['POTCAR']
        if isinstance(list(potcar_settings.values())[-1], dict):
            self.valid_potcars = {key: dct.get('hash' if check_hash else 'symbol') for key, dct in potcar_settings.items()}
        else:
            if check_hash:
                raise ValueError('Cannot check hashes of potcars, since hashes are not included in the entry.')
            self.valid_potcars = potcar_settings
        self.input_set = input_set
        self.check_hash = check_hash
        self.check_potcar = check_potcar

    def get_correction(self, entry: AnyComputedEntry) -> ufloat:
        """
        Args:
            entry (AnyComputedEntry): ComputedEntry or ComputedStructureEntry.

        Raises:
            ValueError: If entry does not contain "potcar_symbols" key.
            CompatibilityError: If entry has wrong potcar hash/symbols.

        Returns:
            ufloat: 0.0 +/- 0.0 (from uncertainties package)
        """
        if SETTINGS.get('PMG_POTCAR_CHECKS') is False or not self.check_potcar:
            return ufloat(0.0, 0.0)
        potcar_spec = entry.parameters.get('potcar_spec')
        if self.check_hash:
            if potcar_spec:
                psp_settings = {dct.get('hash') for dct in potcar_spec if dct}
            else:
                raise ValueError('Cannot check hash without potcar_spec field')
        elif potcar_spec:
            psp_settings = {dct.get('titel').split()[1] for dct in potcar_spec if dct}
        else:
            psp_settings = {sym.split()[1] for sym in entry.parameters['potcar_symbols'] if sym}
        expected_psp = {self.valid_potcars.get(el.symbol) for el in entry.elements}
        if expected_psp != psp_settings:
            raise CompatibilityError(f'Incompatible POTCAR {psp_settings}, expected {expected_psp}')
        return ufloat(0.0, 0.0)

    def __str__(self) -> str:
        return f'{self.input_set.__name__} Potcar Correction'