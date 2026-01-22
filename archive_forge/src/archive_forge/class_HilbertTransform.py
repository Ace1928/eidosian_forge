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
class HilbertTransform(AbivarAble):
    """
    Parameters for the Hilbert-transform method (Screening code)
    i.e. the parameters defining the frequency mesh used for the spectral function
    and the frequency mesh used for the polarizability.
    """

    def __init__(self, nomegasf, domegasf=None, spmeth=1, nfreqre=None, freqremax=None, nfreqim=None, freqremin=None):
        """
        Args:
            nomegasf: Number of points for sampling the spectral function along the real axis.
            domegasf: Step in Ha for the linear mesh used for the spectral function.
            spmeth: Algorithm for the representation of the delta function.
            nfreqre: Number of points along the real axis (linear mesh).
            freqremax: Maximum frequency for W along the real axis (in hartree).
            nfreqim: Number of point along the imaginary axis (Gauss-Legendre mesh).
            freqremin: Minimum frequency for W along the real axis (in hartree).
        """
        self.nomegasf = nomegasf
        self.domegasf = domegasf
        self.spmeth = spmeth
        self.nfreqre = nfreqre
        self.freqremax = freqremax
        self.freqremin = freqremin
        self.nfreqim = nfreqim

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        return {'nomegasf': self.nomegasf, 'domegasf': self.domegasf, 'spmeth': self.spmeth, 'nfreqre': self.nfreqre, 'freqremax': self.freqremax, 'nfreqim': self.nfreqim, 'freqremin': self.freqremin}