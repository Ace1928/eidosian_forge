from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.util.due import Doi, due
def _get_bonds(self, mol):
    """Find all bonds in a molecule.

        Args:
            mol: the molecule. pymatgen Molecule object

        Returns:
            List of tuple. Each tuple correspond to a bond represented by the
            id of the two end atoms.
        """
    n_atoms = len(mol)
    if self.ignore_ionic_bond:
        covalent_atoms = [idx for idx in range(n_atoms) if mol.species[idx].symbol not in self.ionic_element_list]
    else:
        covalent_atoms = list(range(n_atoms))
    all_pairs = list(itertools.combinations(covalent_atoms, 2))
    pair_dists = [mol.get_distance(*p) for p in all_pairs]
    unavailable_elements = set(mol.composition.as_dict()) - set(self.covalent_radius)
    if len(unavailable_elements) > 0:
        raise ValueError(f'The covalent radius for element {unavailable_elements} is not available')
    bond_13 = self.get_13_bonds(self.priority_bonds)
    max_length = [(self.covalent_radius[mol.sites[p[0]].specie.symbol] + self.covalent_radius[mol.sites[p[1]].specie.symbol]) * (1 + (self.priority_cap if p in self.priority_bonds else self.bond_length_cap if p not in bond_13 else self.bond_13_cap)) * (0.1 if self.ignore_halogen_self_bond and p not in self.priority_bonds and (mol.sites[p[0]].specie.symbol in self.halogen_list) and (mol.sites[p[1]].specie.symbol in self.halogen_list) else 1.0) for p in all_pairs]
    return [bond for bond, dist, cap in zip(all_pairs, pair_dists, max_length) if dist <= cap]