from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
class Electrons(AbivarAble, MSONable):
    """The electronic degrees of freedom."""

    def __init__(self, spin_mode='polarized', smearing='fermi_dirac:0.1 eV', algorithm=None, nband=None, fband=None, charge=0.0, comment=None):
        """
        Constructor for Electrons object.

        Args:
            comment: String comment for Electrons
            charge: Total charge of the system. Default is 0.
        """
        super().__init__()
        self.comment = comment
        self.smearing = Smearing.as_smearing(smearing)
        self.spin_mode = SpinMode.as_spinmode(spin_mode)
        self.nband = nband
        self.fband = fband
        self.charge = charge
        self.algorithm = algorithm

    @property
    def nsppol(self):
        """Number of independent spin polarizations."""
        return self.spin_mode.nsppol

    @property
    def nspinor(self):
        """Number of independent spinor components."""
        return self.spin_mode.nspinor

    @property
    def nspden(self):
        """Number of independent density components."""
        return self.spin_mode.nspden

    def as_dict(self):
        """JSON friendly dict representation."""
        dct = {}
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['spin_mode'] = self.spin_mode.as_dict()
        dct['smearing'] = self.smearing.as_dict()
        dct['algorithm'] = self.algorithm.as_dict() if self.algorithm else None
        dct['nband'] = self.nband
        dct['fband'] = self.fband
        dct['charge'] = self.charge
        dct['comment'] = self.comment
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dictionary."""
        dct = dct.copy()
        dct.pop('@module', None)
        dct.pop('@class', None)
        dct['spin_mode'] = MontyDecoder().process_decoded(dct['spin_mode'])
        dct['smearing'] = MontyDecoder().process_decoded(dct['smearing'])
        dct['algorithm'] = MontyDecoder().process_decoded(dct['algorithm']) if dct['algorithm'] else None
        return cls(**dct)

    def to_abivars(self):
        """Return dictionary with Abinit variables."""
        abivars = self.spin_mode.to_abivars()
        abivars.update({'nband': self.nband, 'fband': self.fband, 'charge': self.charge})
        if self.smearing:
            abivars.update(self.smearing.to_abivars())
        if self.algorithm:
            abivars.update(self.algorithm)
        return abivars