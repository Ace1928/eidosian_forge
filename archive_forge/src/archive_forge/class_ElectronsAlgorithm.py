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
class ElectronsAlgorithm(dict, AbivarAble, MSONable):
    """Variables controlling the SCF/NSCF algorithm."""
    _DEFAULT = dict(iprcell=None, iscf=None, diemac=None, diemix=None, diemixmag=None, dielam=None, diegap=None, dielng=None, diecut=None, nstep=50)

    def __init__(self, *args, **kwargs):
        """Initialize object."""
        super().__init__(*args, **kwargs)
        for key in self:
            if key not in self._DEFAULT:
                raise ValueError(f'{type(self).__name__}: No default value has been provided for key={key!r}')

    def to_abivars(self):
        """Dictionary with Abinit input variables."""
        return self.copy()

    def as_dict(self):
        """Convert object to dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, **self.copy()}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build object from dict."""
        dct = dct.copy()
        dct.pop('@module', None)
        dct.pop('@class', None)
        return cls(**dct)