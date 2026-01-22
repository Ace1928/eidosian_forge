from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class EwaldElectrostaticModel(EnergyModel):
    """Wrapper around EwaldSum to calculate the electrostatic energy."""

    def __init__(self, real_space_cut=None, recip_space_cut=None, eta=None, acc_factor=8.0):
        """
        Initializes the model. Args have the same definitions as in
        pymatgen.analysis.ewald.EwaldSummation.

        Args:
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum. Defaults to None,
                which means determine automatically using the formula given
                in gulp 3.1 documentation.
            recip_space_cut (float): Reciprocal space cutoff radius.
                Defaults to None, which means determine automatically using
                the formula given in gulp 3.1 documentation.
            eta (float): Screening parameter. Defaults to None, which means
                determine automatically.
            acc_factor (float): No. of significant figures each sum is
                converged to.
        """
        self.real_space_cut = real_space_cut
        self.recip_space_cut = recip_space_cut
        self.eta = eta
        self.acc_factor = acc_factor

    def get_energy(self, structure: Structure):
        """
        Args:
            structure: Structure

        Returns:
            Energy value
        """
        e = EwaldSummation(structure, real_space_cut=self.real_space_cut, recip_space_cut=self.recip_space_cut, eta=self.eta, acc_factor=self.acc_factor)
        return e.total_energy

    def as_dict(self):
        """MSONable dict"""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {'real_space_cut': self.real_space_cut, 'recip_space_cut': self.recip_space_cut, 'eta': self.eta, 'acc_factor': self.acc_factor}}