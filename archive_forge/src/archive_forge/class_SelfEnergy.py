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
class SelfEnergy(AbivarAble):
    """This object defines the parameters used for the computation of the self-energy."""
    _SIGMA_TYPES = dict(gw=0, hartree_fock=5, sex=6, cohsex=7, model_gw_ppm=8, model_gw_cd=9)
    _SC_MODES = dict(one_shot=0, energy_only=1, wavefunctions=2)

    def __init__(self, se_type, sc_mode, nband, ecutsigx, screening, gw_qprange=1, ppmodel=None, ecuteps=None, ecutwfn=None, gwpara=2):
        """
        Args:
            se_type: Type of self-energy (str)
            sc_mode: Self-consistency mode.
            nband: Number of bands for the Green's function
            ecutsigx: Cutoff energy for the exchange part of the self-energy (Ha units).
            screening: Screening instance.
            gw_qprange: Option for the automatic selection of k-points and bands for GW corrections.
                See Abinit docs for more detail. The default value makes the code computie the
                QP energies for all the point in the IBZ and one band above and one band below the Fermi level.
            ppmodel: PPModel instance with the parameters used for the plasmon-pole technique.
            ecuteps: Cutoff energy for the screening (Ha units).
            ecutwfn: Cutoff energy for the wavefunctions (Default: ecutwfn == ecut).
        """
        if se_type not in self._SIGMA_TYPES:
            raise ValueError(f'SIGMA_TYPE: {se_type} is not supported')
        if sc_mode not in self._SC_MODES:
            raise ValueError(f'Self-consistecy mode {sc_mode} is not supported')
        self.type = se_type
        self.sc_mode = sc_mode
        self.nband = nband
        self.ecutsigx = ecutsigx
        self.screening = screening
        self.gw_qprange = gw_qprange
        self.gwpara = gwpara
        if ppmodel is not None:
            assert screening.use_hilbert is False
            self.ppmodel = PPModel.as_ppmodel(ppmodel)
        self.ecuteps = ecuteps if ecuteps is not None else screening.ecuteps
        self.ecutwfn = ecutwfn
        self.optdriver = 4

    @property
    def use_ppmodel(self):
        """True if we are using the plasmon-pole approximation."""
        return hasattr(self, 'ppmodel')

    @property
    def gwcalctyp(self):
        """Returns the value of the gwcalctyp input variable."""
        dig0 = str(self._SIGMA_TYPES[self.type])
        dig1 = str(self._SC_MODES[self.sc_mode])
        return dig1.strip() + dig0.strip()

    @property
    def symsigma(self):
        """1 if symmetries can be used to reduce the number of q-points."""
        return 1 if self.sc_mode == 'one_shot' else 0

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        abivars = {'gwcalctyp': self.gwcalctyp, 'ecuteps': self.ecuteps, 'ecutsigx': self.ecutsigx, 'symsigma': self.symsigma, 'gw_qprange': self.gw_qprange, 'gwpara': self.gwpara, 'optdriver': self.optdriver, 'nband': self.nband}
        if self.use_ppmodel:
            abivars.update(self.ppmodel.to_abivars())
        return abivars