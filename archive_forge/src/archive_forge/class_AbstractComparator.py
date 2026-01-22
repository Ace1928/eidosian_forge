from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
class AbstractComparator(MSONable, abc.ABC):
    """
    Abstract Comparator class. A Comparator defines how sites are compared in
    a structure.
    """

    @abc.abstractmethod
    def are_equal(self, sp1, sp2) -> bool:
        """
        Defines how the species of two sites are considered equal. For
        example, one can consider sites to have the same species only when
        the species are exactly the same, i.e., Fe2+ matches Fe2+ but not
        Fe3+. Or one can define that only the element matters,
        and all oxidation state information are ignored.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            Boolean indicating whether species are considered equal.
        """
        return False

    @abc.abstractmethod
    def get_hash(self, composition):
        """
        Defines a hash to group structures. This allows structures to be
        grouped efficiently for comparison. The hash must be invariant under
        supercell creation. (e.g. composition is not a good hash, but
        fractional_composition might be). Reduced formula is not a good formula,
        due to weird behavior with fractional occupancy.

        Composition is used here instead of structure because for anonymous
        matches it is much quicker to apply a substitution to a composition
        object than a structure object.

        Args:
            composition (Composition): composition of the structure

        Returns:
            A hashable object. Examples can be string formulas, integers etc.
        """

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation

        Returns:
            Comparator.
        """
        for trans_modules in ['structure_matcher']:
            mod = __import__(f'pymatgen.analysis.{trans_modules}', globals(), locals(), [dct['@class']], 0)
            if hasattr(mod, dct['@class']):
                trans = getattr(mod, dct['@class'])
                return trans()
        raise ValueError('Invalid Comparator dict')

    def as_dict(self):
        """MSONable dict"""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__}