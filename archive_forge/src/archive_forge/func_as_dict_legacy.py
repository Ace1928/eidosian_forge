from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
def as_dict_legacy(self):
    """Returns: MSONable dict."""
    return {'@module': type(self).__module__, '@class': type(self).__name__, 'entries': [entry.as_dict() for entry in self.get_all_entries()], 'working_ion_entry': self.working_ion_entry.as_dict()}