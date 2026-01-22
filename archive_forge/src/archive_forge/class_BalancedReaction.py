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
class BalancedReaction(MSONable):
    """An object representing a complete chemical reaction."""
    TOLERANCE = 1e-06

    @no_type_check
    def __init__(self, reactants_coeffs: Mapping[CompositionLike, int | float], products_coeffs: Mapping[CompositionLike, int | float]) -> None:
        """
        Reactants and products to be specified as dict of {Composition: coeff}.

        Args:
            reactants_coeffs (dict[Composition, float]): Reactants as dict of {Composition: amt}.
            products_coeffs (dict[Composition, float]): Products as dict of {Composition: amt}.
        """
        reactants_coeffs = {Composition(comp): coeff for comp, coeff in reactants_coeffs.items()}
        products_coeffs = {Composition(comp): coeff for comp, coeff in products_coeffs.items()}
        all_reactants = sum((comp * coeff for comp, coeff in reactants_coeffs.items()), Composition())
        all_products = sum((comp * coeff for comp, coeff in products_coeffs.items()), Composition())
        if not all_reactants.almost_equals(all_products, rtol=0, atol=self.TOLERANCE):
            raise ReactionError('Reaction is unbalanced!')
        self.reactants_coeffs: dict = reactants_coeffs
        self.products_coeffs: dict = products_coeffs
        self._coeffs: list[float] = []
        self._els: list[Element | Species] = []
        self._all_comp: list[Composition] = []
        for key in {*reactants_coeffs, *products_coeffs}:
            coeff = products_coeffs.get(key, 0) - reactants_coeffs.get(key, 0)
            if abs(coeff) > self.TOLERANCE:
                self._all_comp += [key]
                self._coeffs += [coeff]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        for comp in self._all_comp:
            coeff2 = other.get_coeff(comp) if comp in other._all_comp else 0
            if abs(self.get_coeff(comp) - coeff2) > self.TOLERANCE:
                return False
        return True

    def __hash__(self) -> int:
        return hash((frozenset(self.reactants_coeffs.items()), frozenset(self.products_coeffs.items())))

    def __str__(self):
        return self._str_from_comp(self._coeffs, self._all_comp)[0]
    __repr__ = __str__

    @overload
    def calculate_energy(self, energies: dict[Composition, ufloat]) -> ufloat:
        pass

    @overload
    def calculate_energy(self, energies: dict[Composition, float]) -> float:
        pass

    def calculate_energy(self, energies):
        """
        Calculates the energy of the reaction.

        Args:
            energies ({Composition: float}): Energy for each composition.
                E.g ., {comp1: energy1, comp2: energy2}.

        Returns:
            reaction energy as a float.
        """
        return sum((amt * energies[c] for amt, c in zip(self._coeffs, self._all_comp)))

    def normalize_to(self, comp: Composition, factor: float=1) -> None:
        """
        Normalizes the reaction to one of the compositions.
        By default, normalizes such that the composition given has a
        coefficient of 1. Another factor can be specified.

        Args:
            comp (Composition): Composition to normalize to
            factor (float): Factor to normalize to. Defaults to 1.
        """
        scale_factor = abs(1 / self._coeffs[self._all_comp.index(comp)] * factor)
        self._coeffs = [c * scale_factor for c in self._coeffs]

    def normalize_to_element(self, element: Species | Element, factor: float=1) -> None:
        """
        Normalizes the reaction to one of the elements.
        By default, normalizes such that the amount of the element is 1.
        Another factor can be specified.

        Args:
            element (Element/Species): Element to normalize to.
            factor (float): Factor to normalize to. Defaults to 1.
        """
        all_comp = self._all_comp
        coeffs = self._coeffs
        current_el_amount = sum((all_comp[i][element] * abs(coeffs[i]) for i in range(len(all_comp)))) / 2
        scale_factor = factor / current_el_amount
        self._coeffs = [c * scale_factor for c in coeffs]

    def get_el_amount(self, element: Element | Species) -> float:
        """
        Returns the amount of the element in the reaction.

        Args:
            element (Element/Species): Element in the reaction

        Returns:
            Amount of that element in the reaction.
        """
        return sum((self._all_comp[i][element] * abs(self._coeffs[i]) for i in range(len(self._all_comp)))) / 2

    @property
    def elements(self) -> list[Element | Species]:
        """List of elements in the reaction."""
        return self._els

    @property
    def coeffs(self) -> list[float]:
        """Final coefficients of the calculated reaction."""
        return self._coeffs[:]

    @property
    def all_comp(self) -> list[Composition]:
        """List of all compositions in the reaction."""
        return self._all_comp

    @property
    def reactants(self) -> list[Composition]:
        """List of reactants."""
        return [self._all_comp[i] for i in range(len(self._all_comp)) if self._coeffs[i] < 0]

    @property
    def products(self) -> list[Composition]:
        """List of products."""
        return [self._all_comp[i] for i in range(len(self._all_comp)) if self._coeffs[i] > 0]

    def get_coeff(self, comp: Composition) -> float:
        """Returns coefficient for a particular composition."""
        return self._coeffs[self._all_comp.index(comp)]

    def normalized_repr_and_factor(self) -> tuple[str, float]:
        """
        Normalized representation for a reaction
        For example, ``4 Li + 2 O -> 2Li2O`` becomes ``2 Li + O -> Li2O``.
        """
        return self._str_from_comp(self._coeffs, self._all_comp, reduce=True)

    @property
    def normalized_repr(self) -> str:
        """
        A normalized representation of the reaction. All factors are converted
        to lowest common factors.
        """
        return self.normalized_repr_and_factor()[0]

    @classmethod
    def _str_from_formulas(cls, coeffs, formulas) -> str:
        reactant_str = []
        product_str = []
        for amt, formula in zip(coeffs, formulas):
            if abs(amt + 1) < cls.TOLERANCE:
                reactant_str.append(formula)
            elif abs(amt - 1) < cls.TOLERANCE:
                product_str.append(formula)
            elif amt < -cls.TOLERANCE:
                reactant_str.append(f'{-amt:.4g} {formula}')
            elif amt > cls.TOLERANCE:
                product_str.append(f'{amt:.4g} {formula}')
        return f'{' + '.join(reactant_str)} -> {' + '.join(product_str)}'

    @classmethod
    def _str_from_comp(cls, coeffs, compositions, reduce=False) -> tuple[str, float]:
        r_coeffs = np.zeros(len(coeffs))
        r_formulas = []
        for idx, (amt, comp) in enumerate(zip(coeffs, compositions)):
            formula, factor = comp.get_reduced_formula_and_factor()
            r_coeffs[idx] = amt * factor
            r_formulas.append(formula)
        if reduce:
            factor = 1 / gcd_float(np.abs(r_coeffs))
            r_coeffs *= factor
        else:
            factor = 1
        return (cls._str_from_formulas(r_coeffs, r_formulas), factor)

    def as_entry(self, energies) -> ComputedEntry:
        """
        Returns a ComputedEntry representation of the reaction.
        """
        relevant_comp = [comp * abs(coeff) for coeff, comp in zip(self._coeffs, self._all_comp)]
        comp: Composition = sum(relevant_comp, Composition())
        entry = ComputedEntry(0.5 * comp, self.calculate_energy(energies))
        entry.name = str(self)
        return entry

    def as_dict(self) -> dict:
        """
        Returns:
            A dictionary representation of BalancedReaction.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'reactants': {str(comp): coeff for comp, coeff in self.reactants_coeffs.items()}, 'products': {str(comp): coeff for comp, coeff in self.products_coeffs.items()}}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): from as_dict().

        Returns:
            A BalancedReaction object.
        """
        reactants = {Composition(comp): coeff for comp, coeff in dct['reactants'].items()}
        products = {Composition(comp): coeff for comp, coeff in dct['products'].items()}
        return cls(reactants, products)

    @classmethod
    def from_str(cls, rxn_str: str) -> Self:
        """
        Generates a balanced reaction from a string. The reaction must
        already be balanced.

        Args:
            rxn_string (str): The reaction string. For example, "4 Li + O2 -> 2Li2O"

        Returns:
            BalancedReaction
        """
        rct_str, prod_str = rxn_str.split('->')

        def get_comp_amt(comp_str):
            return {Composition(m.group(2)): float(m.group(1) or 1) for m in re.finditer('([\\d\\.]*(?:[eE]-?[\\d\\.]+)?)\\s*([A-Z][\\w\\.\\(\\)]*)', comp_str)}
        return BalancedReaction(get_comp_amt(rct_str), get_comp_amt(prod_str))