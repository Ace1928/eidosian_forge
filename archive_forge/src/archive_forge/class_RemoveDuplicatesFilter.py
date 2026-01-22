from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class RemoveDuplicatesFilter(AbstractStructureFilter):
    """This filter removes exact duplicate structures from the transmuter."""

    def __init__(self, structure_matcher: dict | StructureMatcher | None=None, symprec: float | None=None) -> None:
        """Remove duplicate structures based on the structure matcher
        and symmetry (if symprec is given).

        Args:
            structure_matcher (dict | StructureMatcher, optional): Provides a structure matcher to be used for
                structure comparison.
            symprec (float, optional): The precision in the symmetry finder algorithm if None (
                default value), no symmetry check is performed and only the
                structure matcher is used. A recommended value is 1e-5.
        """
        self.symprec = symprec
        self.structure_list: dict[str, list[Structure]] = defaultdict(list)
        if not isinstance(structure_matcher, (dict, StructureMatcher, type(None))):
            raise ValueError(f'structure_matcher={structure_matcher!r} must be a dict, StructureMatcher or None')
        if isinstance(structure_matcher, dict):
            self.structure_matcher = StructureMatcher.from_dict(structure_matcher)
        else:
            self.structure_matcher = structure_matcher or StructureMatcher(comparator=ElementComparator())

    def test(self, structure: Structure) -> bool:
        """
        Args:
            structure (Structure): Input structure to test.

        Returns:
            bool: True if structure is not in list.
        """
        hash_comp = self.structure_matcher._comparator.get_hash(structure.composition)
        if not self.structure_list[hash_comp]:
            self.structure_list[hash_comp].append(structure)
            return True

        def get_spg_num(struct: Structure) -> int:
            finder = SpacegroupAnalyzer(struct, symprec=self.symprec)
            return finder.get_space_group_number()
        for struct in self.structure_list[hash_comp]:
            if (self.symprec is None or get_spg_num(struct) == get_spg_num(structure)) and self.structure_matcher.fit(struct, structure):
                return False
        self.structure_list[hash_comp].append(structure)
        return True