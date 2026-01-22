from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
class ComputedReaction(Reaction):
    """
    Convenience class to generate a reaction from ComputedEntry objects, with
    some additional attributes, such as a reaction energy based on computed
    energies.
    """

    def __init__(self, reactant_entries: list[ComputedEntry], product_entries: list[ComputedEntry]) -> None:
        """
        Args:
            reactant_entries ([ComputedEntry]): List of reactant_entries.
            product_entries ([ComputedEntry]): List of product_entries.
        """
        self._reactant_entries = reactant_entries
        self._product_entries = product_entries
        self._all_entries = reactant_entries + product_entries
        reactant_comp = [entry.composition.reduced_composition for entry in reactant_entries]
        product_comp = [entry.composition.reduced_composition for entry in product_entries]
        super().__init__(list(reactant_comp), list(product_comp))

    @property
    def all_entries(self):
        """
        Equivalent of all_comp but returns entries, in the same order as the
        coefficients.
        """
        entries = []
        for comp in self._all_comp:
            for entry in self._all_entries:
                if entry.reduced_formula == comp.reduced_formula:
                    entries.append(entry)
                    break
        return entries

    @property
    def calculated_reaction_energy(self) -> float:
        """
        Returns:
            float: The calculated reaction energy.
        """
        calc_energies: dict[Composition, float] = {}
        for entry in self._reactant_entries + self._product_entries:
            comp, factor = entry.composition.get_reduced_composition_and_factor()
            calc_energies[comp] = min(calc_energies.get(comp, float('inf')), entry.energy / factor)
        return self.calculate_energy(calc_energies)

    @property
    def calculated_reaction_energy_uncertainty(self) -> float:
        """
        Calculates the uncertainty in the reaction energy based on the uncertainty in the
        energies of the products and reactants.
        """
        calc_energies: dict[Composition, float] = {}
        for entry in self._reactant_entries + self._product_entries:
            comp, factor = entry.composition.get_reduced_composition_and_factor()
            energy_ufloat = ufloat(entry.energy, entry.correction_uncertainty)
            calc_energies[comp] = min(calc_energies.get(comp, float('inf')), energy_ufloat / factor)
        return self.calculate_energy(calc_energies).std_dev

    def as_dict(self) -> dict:
        """
        Returns:
            A dictionary representation of ComputedReaction.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'reactants': [entry.as_dict() for entry in self._reactant_entries], 'products': [entry.as_dict() for entry in self._product_entries]}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): from as_dict().

        Returns:
            A ComputedReaction object.
        """
        reactants = [MontyDecoder().process_decoded(entry) for entry in dct['reactants']]
        products = [MontyDecoder().process_decoded(entry) for entry in dct['products']]
        return cls(reactants, products)