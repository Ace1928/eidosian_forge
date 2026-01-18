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
def test_nvtberendsen(asap3, equilibrated, berendsenparams, allraise):
    t, _ = propagate(Atoms(equilibrated), asap3, NVTBerendsen, berendsenparams['nvt'])
    assert abs(t - berendsenparams['nvt']['temperature_K']) < 0.5