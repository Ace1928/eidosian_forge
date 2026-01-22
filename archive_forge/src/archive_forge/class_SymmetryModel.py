from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class SymmetryModel(EnergyModel):
    """
    Sets the energy to the negative of the spacegroup number. Higher symmetry =>
    lower "energy".

    Args have same meaning as in pymatgen.symmetry.SpacegroupAnalyzer.
    """

    def __init__(self, symprec: float=0.1, angle_tolerance=5):
        """
        Args:
            symprec (float): Symmetry tolerance. Defaults to 0.1.
            angle_tolerance (float): Tolerance for angles. Defaults to 5 degrees.
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_energy(self, structure: Structure):
        """
        Args:
            structure: Structure

        Returns:
            Energy value
        """
        spg_analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance)
        return -spg_analyzer.get_space_group_number()

    def as_dict(self):
        """MSONable dict"""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {'symprec': self.symprec, 'angle_tolerance': self.angle_tolerance}}