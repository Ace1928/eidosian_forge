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
class PPModel(AbivarAble, MSONable):
    """
    Parameters defining the plasmon-pole technique.
    The common way to instantiate a PPModel object is via the class method PPModel.as_ppmodel(string).
    """

    @classmethod
    def as_ppmodel(cls, obj):
        """
        Constructs an instance of PPModel from obj.

        Accepts obj in the form:
            * PPmodel instance
            * string. e.g "godby:12.3 eV", "linden".
        """
        if isinstance(obj, cls):
            return obj
        if ':' not in obj:
            mode, plasmon_freq = (obj, None)
        else:
            mode, plasmon_freq = obj.split(':')
            try:
                plasmon_freq = float(plasmon_freq)
            except ValueError:
                plasmon_freq, unit = plasmon_freq.split()
                plasmon_freq = units.Energy(float(plasmon_freq), unit).to('Ha')
        return cls(mode=mode, plasmon_freq=plasmon_freq)

    def __init__(self, mode='godby', plasmon_freq=None):
        """
        Args:
            mode: ppmodel type
            plasmon_freq: Plasmon frequency in Ha.
        """
        if isinstance(mode, str):
            mode = PPModelModes[mode]
        self.mode = mode
        self.plasmon_freq = plasmon_freq

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('mode', 'plasmon_freq')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        other = cast(PPModel, other)
        if self.mode != other.mode:
            return False
        if self.plasmon_freq is None:
            return other.plasmon_freq is None
        return np.allclose(self.plasmon_freq, other.plasmon_freq)

    def __bool__(self):
        return self.mode != PPModelModes.noppmodel

    def __repr__(self):
        return f'<{type(self).__name__} at {id(self)}, mode = {self.mode}>'

    def to_abivars(self):
        """Return dictionary with Abinit variables."""
        if self:
            return {'ppmodel': self.mode.value, 'ppmfrq': self.plasmon_freq}
        return {}

    @classmethod
    def get_noppmodel(cls):
        """Calculation without plasmon-pole model."""
        return cls(mode='noppmodel', plasmon_freq=None)

    def as_dict(self):
        """Convert object to dictionary."""
        return {'mode': self.mode.name, 'plasmon_freq': self.plasmon_freq, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dictionary."""
        return cls(mode=dct['mode'], plasmon_freq=dct['plasmon_freq'])