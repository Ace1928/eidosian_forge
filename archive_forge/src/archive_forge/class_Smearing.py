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
class Smearing(AbivarAble, MSONable):
    """
    Variables defining the smearing technique. The preferred way to instantiate
    a `Smearing` object is via the class method Smearing.as_smearing(string).
    """
    _mode2occopt = dict(nosmearing=1, fermi_dirac=3, marzari4=4, marzari5=5, methfessel=6, gaussian=7)

    def __init__(self, occopt, tsmear):
        """Build object with occopt and tsmear."""
        self.occopt = occopt
        self.tsmear = tsmear

    def __str__(self):
        string = f'occopt {self.occopt} # {self.mode} Smearing\n'
        if self.tsmear:
            string += f'tsmear {self.tsmear}'
        return string

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('occopt', 'tsmear')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        other = cast(Smearing, other)
        return self.occopt == other.occopt and np.allclose(self.tsmear, other.tsmear)

    def __bool__(self):
        return self.mode != 'nosmearing'

    @classmethod
    def as_smearing(cls, obj):
        """
        Constructs an instance of `Smearing` from obj. Accepts obj in the form:

            * Smearing instance
            * "name:tsmear"  e.g. "gaussian:0.004"  (Hartree units)
            * "name:tsmear units" e.g. "gaussian:0.1 eV"
            * None --> no smearing
        """
        if obj is None:
            return Smearing.nosmearing()
        if isinstance(obj, cls):
            return obj
        if obj == 'nosmearing':
            return cls.nosmearing()
        obj, tsmear = obj.split(':')
        obj.strip()
        occopt = cls._mode2occopt[obj]
        try:
            tsmear = float(tsmear)
        except ValueError:
            tsmear, unit = tsmear.split()
            tsmear = units.Energy(float(tsmear), unit).to('Ha')
        return cls(occopt, tsmear)

    @property
    def mode(self):
        """String with smearing technique."""
        for mode_str, occopt in self._mode2occopt.items():
            if occopt == self.occopt:
                return mode_str
        raise AttributeError(f'Unknown occopt {self.occopt}')

    @staticmethod
    def nosmearing():
        """Build object for calculations without smearing."""
        return Smearing(1, 0.0)

    def to_abivars(self):
        """Return dictionary with Abinit variables."""
        if self.mode == 'nosmearing':
            return {'occopt': 1, 'tsmear': 0.0}
        return {'occopt': self.occopt, 'tsmear': self.tsmear}

    def as_dict(self):
        """JSON-friendly dict representation of Smearing."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'occopt': self.occopt, 'tsmear': self.tsmear}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dict."""
        return cls(dct['occopt'], dct['tsmear'])