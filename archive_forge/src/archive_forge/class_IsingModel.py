from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class IsingModel(EnergyModel):
    """A very simple Ising model, with r^2 decay."""

    def __init__(self, j, max_radius):
        """
        Args:
            j (float): The interaction parameter. E = J * spin1 * spin2.
            radius (float): max_radius for the interaction.
        """
        self.j = j
        self.max_radius = max_radius

    def get_energy(self, structure: Structure):
        """
        Args:
            structure: Structure

        Returns:
            Energy value
        """
        all_nn = structure.get_all_neighbors(r=self.max_radius)
        energy = 0
        for idx, nns in enumerate(all_nn):
            s1 = getattr(structure[idx].specie, 'spin', 0)
            for nn in nns:
                energy += self.j * s1 * getattr(nn.specie, 'spin', 0) / nn.nn_distance ** 2
        return energy

    def as_dict(self):
        """MSONable dict"""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {'j': self.j, 'max_radius': self.max_radius}}