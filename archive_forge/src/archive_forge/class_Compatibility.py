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
class Compatibility(MSONable, abc.ABC):
    """Abstract Compatibility class, not intended for direct use.
    Compatibility classes are used to correct the energies of an entry or a set
    of entries. All Compatibility classes must implement get_adjustments() method.
    """

    @abc.abstractmethod
    def get_adjustments(self, entry: AnyComputedEntry) -> list[EnergyAdjustment]:
        """Get the energy adjustments for a ComputedEntry.

        This method must generate a list of EnergyAdjustment objects
        of the appropriate type (constant, composition-based, or temperature-based)
        to be applied to the ComputedEntry, and must raise a CompatibilityError
        if the entry is not compatible.

        Args:
            entry: A ComputedEntry object.

        Returns:
            list[EnergyAdjustment]: A list of EnergyAdjustment to be applied to the
                Entry.

        Raises:
            CompatibilityError if the entry is not compatible
        """
        raise NotImplementedError

    def process_entry(self, entry: ComputedEntry, **kwargs) -> ComputedEntry | None:
        """Process a single entry with the chosen Corrections. Note
        that this method will change the data of the original entry.

        Args:
            entry: A ComputedEntry object.
            **kwargs: Will be passed to process_entries().

        Returns:
            An adjusted entry if entry is compatible, else None.
        """
        try:
            return self.process_entries(entry, **kwargs)[0]
        except IndexError:
            return None

    def process_entries(self, entries: AnyComputedEntry | list[AnyComputedEntry], clean: bool=True, verbose: bool=False, inplace: bool=True, on_error: Literal['ignore', 'warn', 'raise']='ignore') -> list[AnyComputedEntry]:
        """Process a sequence of entries with the chosen Compatibility scheme.

        Warning: This method changes entries in place! All changes can be undone and original entries
        restored by setting entry.energy_adjustments = [].

        Args:
            entries (AnyComputedEntry | list[AnyComputedEntry]): A sequence of
                Computed(Structure)Entry objects.
            clean (bool): Whether to remove any previously-applied energy adjustments.
                If True, all EnergyAdjustment are removed prior to processing the Entry.
                Defaults to True.
            verbose (bool): Whether to display progress bar for processing multiple entries.
                Defaults to False.
            inplace (bool): Whether to adjust input entries in place. Defaults to True.
            on_error ('ignore' | 'warn' | 'raise'): What to do when get_adjustments(entry)
                raises CompatibilityError. Defaults to 'ignore'.

        Returns:
            list[AnyComputedEntry]: Adjusted entries. Entries in the original list incompatible with
                chosen correction scheme are excluded from the returned list.
        """
        if isinstance(entries, ComputedEntry):
            entries = [entries]
        processed_entry_list: list[AnyComputedEntry] = []
        if not inplace:
            entries = copy.deepcopy(entries)
        for entry in tqdm(entries, disable=not verbose):
            ignore_entry = False
            if clean:
                entry.energy_adjustments = []
            try:
                adjustments = self.get_adjustments(entry)
            except CompatibilityError as exc:
                if on_error == 'raise':
                    raise exc
                if on_error == 'warn':
                    warnings.warn(str(exc))
                continue
            for ea in adjustments:
                if (ea.name, ea.cls, ea.value) in [(ea2.name, ea2.cls, ea2.value) for ea2 in entry.energy_adjustments]:
                    pass
                elif (ea.name, ea.cls) in [(ea2.name, ea2.cls) for ea2 in entry.energy_adjustments]:
                    ignore_entry = True
                    warnings.warn(f'Entry {entry.entry_id} already has an energy adjustment called {ea.name}, but its value differs from the value of {ea.value:.3f} calculated here. This Entry will be discarded.')
                else:
                    entry.energy_adjustments.append(ea)
            if not ignore_entry:
                processed_entry_list.append(entry)
        return processed_entry_list

    @staticmethod
    def explain(entry):
        """Prints an explanation of the energy adjustments applied by the
        Compatibility class. Inspired by the "explain" methods in many database
        methodologies.

        Args:
            entry: A ComputedEntry.
        """
        print(f'The uncorrected energy of {entry.composition} is {entry.uncorrected_energy:.3f} eV ({entry.uncorrected_energy / entry.composition.num_atoms:.3f} eV/atom).')
        if len(entry.energy_adjustments) > 0:
            print('The following energy adjustments have been applied to this entry:')
            for adj in entry.energy_adjustments:
                print(f'\t\t{adj.name}: {adj.value:.3f} eV ({adj.value / entry.composition.num_atoms:.3f} eV/atom)')
        elif entry.correction == 0:
            print('No energy adjustments have been applied to this entry.')
        print(f'The final energy after adjustments is {entry.energy:.3f} eV ({entry.energy_per_atom:.3f} eV/atom).')