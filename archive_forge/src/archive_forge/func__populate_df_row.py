from __future__ import annotations
import copy
import os
import warnings
from itertools import groupby
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import (
from pymatgen.entries.computed_entries import ComputedStructureEntry, ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
def _populate_df_row(self, struct_group, comp, sg, n, pd_type_1, pd_type_2, all_entries):
    """Helper function to populate a row of the mixing state DataFrame, given
        a list of matched structures.
        """
    entries_type_1 = sorted((e for e in all_entries if e.entry_id in [s.entry_id for s in struct_group] and e.parameters['run_type'] in self.valid_rtypes_1), key=lambda x: x.energy_per_atom)
    first_entry = entries_type_1[0] if len(entries_type_1) > 0 else None
    entries_type_2 = sorted((e for e in all_entries if e.entry_id in [s.entry_id for s in struct_group] and e.parameters['run_type'] in self.valid_rtypes_2), key=lambda x: x.energy_per_atom)
    second_entry = entries_type_2[0] if len(entries_type_2) > 0 else None
    stable_1 = False
    id1 = first_entry.entry_id if first_entry else None
    id2 = second_entry.entry_id if second_entry else None
    rt1 = first_entry.parameters['run_type'] if first_entry else None
    rt2 = second_entry.parameters['run_type'] if second_entry else None
    energy_1 = first_entry.energy_per_atom if first_entry else np.nan
    energy_2 = second_entry.energy_per_atom if second_entry else np.nan
    if pd_type_1:
        stable_1 = first_entry in pd_type_1.stable_entries
    hull_energy_1, hull_energy_2 = (np.nan, np.nan)
    if pd_type_1:
        hull_energy_1 = pd_type_1.get_hull_energy_per_atom(comp)
    if pd_type_2:
        hull_energy_2 = pd_type_2.get_hull_energy_per_atom(comp)
    return [comp.reduced_formula, sg, n, stable_1, id1, id2, rt1, rt2, energy_1, energy_2, hull_energy_1, hull_energy_2]