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
def to_abivars(self):
    """Returns a dictionary with the abinit variables."""
    abivars = {'bs_calctype': 1, 'bs_loband': self.bs_loband, 'mbpt_sciss': self.mbpt_sciss, 'ecuteps': self.ecuteps, 'bs_algorithm': self._ALGO2VAR[self.algo], 'bs_coulomb_term': 21, 'mdf_epsinf': self.mdf_epsinf, 'bs_exchange_term': 1 if self.with_lf else 0, 'inclvkb': self.inclvkb, 'zcut': self.zcut, 'bs_freq_mesh': self.bs_freq_mesh, 'bs_coupling': self._EXC_TYPES[self.exc_type], 'optdriver': self.optdriver}
    if self.use_haydock:
        abivars.update(bs_haydock_niter=100, bs_hayd_term=0, bs_haydock_tol=[0.05, 0])
    elif self.use_direct_diago or self.use_cg:
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown algorithm for EXC: {self.algo}')
    abivars.update(self.kwargs)
    return abivars