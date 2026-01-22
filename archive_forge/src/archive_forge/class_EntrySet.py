from __future__ import annotations
import collections
import csv
import datetime
import itertools
import json
import logging
import multiprocessing as mp
import re
from typing import TYPE_CHECKING, Literal
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Composition, Element
class EntrySet(collections.abc.MutableSet, MSONable):
    """A convenient container for manipulating entries. Allows for generating
    subsets, dumping into files, etc.
    """

    def __init__(self, entries: Iterable[PDEntry | ComputedEntry | ComputedStructureEntry]):
        """
        Args:
            entries: All the entries.
        """
        self.entries = set(entries)

    def __contains__(self, item):
        return item in self.entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def add(self, element):
        """Add an entry.

        Args:
            element: Entry
        """
        self.entries.add(element)

    def discard(self, element):
        """Discard an entry.

        Args:
            element: Entry
        """
        self.entries.discard(element)

    @property
    def chemsys(self) -> set:
        """
        Returns:
            set representing the chemical system, e.g., {"Li", "Fe", "P", "O"}.
        """
        chemsys = set()
        for e in self.entries:
            chemsys.update([el.symbol for el in e.composition])
        return chemsys

    @property
    def ground_states(self) -> set:
        """A set containing only the entries that are ground states, i.e., the lowest energy
        per atom entry at each composition.
        """
        entries = sorted(self.entries, key=lambda e: e.reduced_formula)
        return {min(g, key=lambda e: e.energy_per_atom) for _, g in itertools.groupby(entries, key=lambda e: e.reduced_formula)}

    def remove_non_ground_states(self):
        """Removes all non-ground state entries, i.e., only keep the lowest energy
        per atom entry at each composition.
        """
        self.entries = self.ground_states

    def is_ground_state(self, entry) -> bool:
        """Boolean indicating whether a given Entry is a ground state."""
        return entry in self.ground_states

    def get_subset_in_chemsys(self, chemsys: list[str]):
        """Returns an EntrySet containing only the set of entries belonging to
        a particular chemical system (in this definition, it includes all sub
        systems). For example, if the entries are from the
        Li-Fe-P-O system, and chemsys=["Li", "O"], only the Li, O,
        and Li-O entries are returned.

        Args:
            chemsys: Chemical system specified as list of elements. E.g.,
                ["Li", "O"]

        Returns:
            EntrySet
        """
        chem_sys = set(chemsys)
        if not chem_sys.issubset(self.chemsys):
            raise ValueError(f'{sorted(chem_sys)} is not a subset of {sorted(self.chemsys)}, extra: {chem_sys - self.chemsys}')
        subset = set()
        for e in self.entries:
            elements = [sp.symbol for sp in e.composition]
            if chem_sys.issuperset(elements):
                subset.add(e)
        return EntrySet(subset)

    def as_dict(self) -> dict[Literal['entries'], list[Entry]]:
        """Returns MSONable dict."""
        return {'entries': list(self.entries)}

    def to_csv(self, filename: str, latexify_names: bool=False) -> None:
        """Exports PDEntries to a csv.

        Args:
            filename: Filename to write to.
            entries: PDEntries to export.
            latexify_names: Format entry names to be LaTex compatible,
                e.g., Li_{2}O
        """
        els: set[Element] = set()
        for entry in self.entries:
            els.update(entry.elements)
        elements = sorted(els, key=lambda a: a.X)
        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Name'] + [el.symbol for el in elements] + ['Energy'])
            for entry in self.entries:
                row: list[str] = [entry.name if not latexify_names else re.sub('([0-9]+)', '_{\\1}', entry.name)]
                row.extend([str(entry.composition[el]) for el in elements])
                row.append(str(entry.energy))
                writer.writerow(row)

    @classmethod
    def from_csv(cls, filename: str) -> Self:
        """Imports PDEntries from a csv.

        Args:
            filename: Filename to import from.

        Returns:
            List of Elements, List of PDEntries
        """
        with open(filename, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            entries = []
            header_read = False
            elements: list[str] = []
            for row in reader:
                if not header_read:
                    elements = row[1:len(row) - 1]
                    header_read = True
                else:
                    name = row[0]
                    energy = float(row[-1])
                    comp = {}
                    for ind in range(1, len(row) - 1):
                        if float(row[ind]) > 0:
                            comp[Element(elements[ind - 1])] = float(row[ind])
                    entries.append(PDEntry(Composition(comp), energy, name))
        return cls(entries)