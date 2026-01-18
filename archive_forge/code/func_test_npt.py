import pytest
from ase import Atoms
from ase.units import fs, GPa, bar
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
@pytest.mark.slow
def test_npt(asap3, equilibrated, berendsenparams, allraise):
    params = berendsenparams['npt']
    t, p = propagate(Atoms(equilibrated), asap3, NPT, dict(temperature_K=params['temperature_K'], externalstress=params['pressure_au'], ttime=params['taut'], pfactor=params['taup'] ** 2 * 1.3))
    n = len(equilibrated)
    assert abs(t - (n - 1) / n * berendsenparams['npt']['temperature_K']) < 1.0
    assert abs(p - berendsenparams['npt']['pressure_au']) < 100.0 * bar