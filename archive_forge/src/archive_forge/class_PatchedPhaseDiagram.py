from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
class PatchedPhaseDiagram(PhaseDiagram):
    """
    Computing the Convex Hull of a large set of data in multiple dimensions is
    highly expensive. This class acts to breakdown large chemical spaces into
    smaller chemical spaces which can be computed much more quickly due to having
    both reduced dimensionality and data set sizes.

    Attributes:
        subspaces ({str: {Element, }}): Dictionary of the sets of elements for each of the
            PhaseDiagrams within the PatchedPhaseDiagram.
        pds ({str: PhaseDiagram}): Dictionary of PhaseDiagrams within the
            PatchedPhaseDiagram.
        all_entries (list[PDEntry]): All entries provided for Phase Diagram construction.
            Note that this does not mean that all these entries are actually used in
            the phase diagram. For example, this includes the positive formation energy
            entries that are filtered out before Phase Diagram construction.
        min_entries (list[PDEntry]): List of the  lowest energy entries for each composition
            in the data provided for Phase Diagram construction.
        el_refs (list[PDEntry]): List of elemental references for the phase diagrams.
            These are entries corresponding to the lowest energy element entries for
            simple compositional phase diagrams.
        elements (list[Element]): List of elements in the phase diagram.
    """

    def __init__(self, entries: Sequence[PDEntry] | set[PDEntry], elements: Sequence[Element] | None=None, keep_all_spaces: bool=False, verbose: bool=False) -> None:
        """
        Args:
            entries (list[PDEntry]): A list of PDEntry-like objects having an
                energy, energy_per_atom and composition.
            elements (list[Element], optional): Optional list of elements in the phase
                diagram. If set to None, the elements are determined from
                the entries themselves and are sorted alphabetically.
                If specified, element ordering (e.g. for pd coordinates)
                is preserved.
            keep_all_spaces (bool): Boolean control on whether to keep chemical spaces
                that are subspaces of other spaces.
            verbose (bool): Whether to show progress bar during convex hull construction.
        """
        if elements is None:
            elements = sorted({els for entry in entries for els in entry.elements})
        self.dim = len(elements)
        entries = sorted(entries, key=lambda e: e.composition.reduced_composition)
        el_refs: dict[Element, PDEntry] = {}
        min_entries = []
        all_entries: list[PDEntry] = []
        for composition, group_iter in itertools.groupby(entries, key=lambda e: e.composition.reduced_composition):
            group = list(group_iter)
            min_entry = min(group, key=lambda e: e.energy_per_atom)
            if composition.is_element:
                el_refs[composition.elements[0]] = min_entry
            min_entries.append(min_entry)
            all_entries.extend(group)
        if len(el_refs) < self.dim:
            missing = set(elements) - set(el_refs)
            raise ValueError(f'Missing terminal entries for elements {sorted(map(str, missing))}')
        if len(el_refs) > self.dim:
            extra = set(el_refs) - set(elements)
            raise ValueError(f'There are more terminal elements than dimensions: {extra}')
        data = np.array([[*(entry.composition.get_atomic_fraction(el) for el in elements), entry.energy_per_atom] for entry in min_entries])
        vec = [el_refs[el].energy_per_atom for el in elements] + [-1]
        form_e = -np.dot(data, vec)
        inds = np.where(form_e < -PhaseDiagram.formation_energy_tol)[0].tolist()
        inds.extend([min_entries.index(el) for el in el_refs.values()])
        self.qhull_entries = tuple((min_entries[idx] for idx in inds))
        self._qhull_spaces = tuple((frozenset(entry.elements) for entry in self.qhull_entries))
        spaces = {s for s in self._qhull_spaces if len(s) > 1}
        if not keep_all_spaces and len(spaces) > 1:
            max_size = max((len(s) for s in spaces))
            systems = set()
            for idx in range(2, max_size + 1):
                test = (s for s in spaces if len(s) == idx)
                refer = (s for s in spaces if len(s) > idx)
                systems |= {t for t in test if not any((t.issubset(r) for r in refer))}
            spaces = systems
        self.spaces = sorted(spaces, key=len, reverse=False)
        self.pds = dict((self._get_pd_patch_for_space(s) for s in tqdm(self.spaces, disable=not verbose)))
        self.all_entries = all_entries
        self.el_refs = el_refs
        self.elements = elements
        _stable_entries = {se for pd in self.pds.values() for se in pd._stable_entries}
        self._stable_entries = tuple(_stable_entries | {*self.el_refs.values()})
        self._stable_spaces = tuple((frozenset(entry.elements) for entry in self._stable_entries))

    def __repr__(self):
        return f'{type(self).__name__} covering {len(self.spaces)} sub-spaces'

    def __len__(self):
        return len(self.spaces)

    def __getitem__(self, item: frozenset[Element]) -> PhaseDiagram:
        return self.pds[item]

    def __setitem__(self, key: frozenset[Element], value: PhaseDiagram) -> None:
        self.pds[key] = value

    def __delitem__(self, key: frozenset[Element]) -> None:
        del self.pds[key]

    def __iter__(self) -> Iterator[PhaseDiagram]:
        return iter(self.pds.values())

    def __contains__(self, item: frozenset[Element]) -> bool:
        return item in self.pds

    def as_dict(self) -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]: MSONable dictionary representation of PatchedPhaseDiagram.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'all_entries': [entry.as_dict() for entry in self.all_entries], 'elements': [entry.as_dict() for entry in self.elements]}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): dictionary representation of PatchedPhaseDiagram.

        Returns:
            PatchedPhaseDiagram
        """
        entries = [MontyDecoder().process_decoded(entry) for entry in dct['all_entries']]
        elements = [Element.from_dict(elem) for elem in dct['elements']]
        return cls(entries, elements)

    def get_pd_for_entry(self, entry: Entry | Composition) -> PhaseDiagram:
        """
        Get the possible phase diagrams for an entry.

        Args:
            entry (PDEntry | Composition): A PDEntry or Composition-like object

        Returns:
            PhaseDiagram: phase diagram that the entry is part of

        Raises:
            ValueError: If no suitable PhaseDiagram is found for the entry.
        """
        entry_space = frozenset(entry.elements) if isinstance(entry, Composition) else frozenset(entry.elements)
        try:
            return self.pds[entry_space]
        except KeyError:
            for space, pd in self.pds.items():
                if space.issuperset(entry_space):
                    return pd
        raise ValueError(f'No suitable PhaseDiagrams found for {entry}.')

    def get_decomposition(self, comp: Composition) -> dict[PDEntry, float]:
        """
        See PhaseDiagram.

        Args:
            comp (Composition): A composition

        Returns:
            Decomposition as a dict of {PDEntry: amount} where amount
            is the amount of the fractional composition.
        """
        try:
            pd = self.get_pd_for_entry(comp)
            return pd.get_decomposition(comp)
        except ValueError as exc:
            warnings.warn(f'{exc} Using SLSQP to find decomposition')
            competing_entries = self._get_stable_entries_in_space(frozenset(comp.elements))
            return _get_slsqp_decomp(comp, competing_entries)

    def get_equilibrium_reaction_energy(self, entry: Entry) -> float:
        """
        See PhaseDiagram.

        NOTE this is only approximately the same as the what we would get
        from `PhaseDiagram` as we make use of the slsqp approach inside
        get_phase_separation_energy().

        Args:
            entry (PDEntry): A PDEntry like object

        Returns:
            Equilibrium reaction energy of entry. Stable entries should have
            equilibrium reaction energy <= 0. The energy is given per atom.
        """
        return self.get_phase_separation_energy(entry, stable_only=True)

    def get_decomp_and_e_above_hull(self, entry: PDEntry, allow_negative: bool=False, check_stable: bool=False, on_error: Literal['raise', 'warn', 'ignore']='raise') -> tuple[dict[PDEntry, float], float] | tuple[None, None]:
        """Same as method on parent class PhaseDiagram except check_stable defaults to False
        for speed. See https://github.com/materialsproject/pymatgen/issues/2840 for details.
        """
        return super().get_decomp_and_e_above_hull(entry=entry, allow_negative=allow_negative, check_stable=check_stable, on_error=on_error)

    def _get_facet_and_simplex(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('_get_facet_and_simplex() not implemented for PatchedPhaseDiagram')

    def _get_all_facets_and_simplexes(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('_get_all_facets_and_simplexes() not implemented for PatchedPhaseDiagram')

    def _get_facet_chempots(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('_get_facet_chempots() not implemented for PatchedPhaseDiagram')

    def _get_simplex_intersections(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('_get_simplex_intersections() not implemented for PatchedPhaseDiagram')

    def get_composition_chempots(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_composition_chempots() not implemented for PatchedPhaseDiagram')

    def get_all_chempots(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_all_chempots() not implemented for PatchedPhaseDiagram')

    def get_transition_chempots(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_transition_chempots() not implemented for PatchedPhaseDiagram')

    def get_critical_compositions(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_critical_compositions() not implemented for PatchedPhaseDiagram')

    def get_element_profile(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_element_profile() not implemented for PatchedPhaseDiagram')

    def get_chempot_range_map(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_chempot_range_map() not implemented for PatchedPhaseDiagram')

    def getmu_vertices_stability_phase(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('getmu_vertices_stability_phase() not implemented for PatchedPhaseDiagram')

    def get_chempot_range_stability_phase(self):
        """Not Implemented - See PhaseDiagram."""
        raise NotImplementedError('get_chempot_range_stability_phase() not implemented for PatchedPhaseDiagram')

    def _get_pd_patch_for_space(self, space: frozenset[Element]) -> tuple[frozenset[Element], PhaseDiagram]:
        """
        Args:
            space (frozenset[Element]): chemical space of the form A-B-X.

        Returns:
            space, PhaseDiagram for the given chemical space
        """
        space_entries = [e for e, s in zip(self.qhull_entries, self._qhull_spaces) if space.issuperset(s)]
        return (space, PhaseDiagram(space_entries))