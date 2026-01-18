import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
def test_vibrations_methods(self, testdir, random_dimer):
    vib = Vibrations(random_dimer)
    vib.run()
    vib_energies = vib.get_energies()
    for image in vib.iterimages():
        assert len(image) == 2
    thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear', atoms=vib.atoms, symmetrynumber=2, spin=0)
    thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.0, verbose=False)
    with open(self.logfile, 'w') as fd:
        vib.summary(log=fd)
    with open(self.logfile, 'rt') as fd:
        log_txt = fd.read()
        assert log_txt == '\n'.join(VibrationsData._tabulate_from_energies(vib_energies)) + '\n'
    last_mode = vib.get_mode(-1)
    scale = 0.5
    assert_array_almost_equal(vib.show_as_force(-1, scale=scale, show=False).get_forces(), last_mode * 3 * len(vib.atoms) * scale)
    vib.write_mode(n=3, nimages=5)
    for i in range(3):
        assert not Path('vib.{}.traj'.format(i)).is_file()
    mode_traj = ase.io.read('vib.3.traj', index=':')
    assert len(mode_traj) == 5
    assert_array_almost_equal(mode_traj[0].get_all_distances(), random_dimer.get_all_distances())
    with pytest.raises(AssertionError):
        assert_array_almost_equal(mode_traj[4].get_all_distances(), random_dimer.get_all_distances())
    assert vib.clean(empty_files=True) == 0
    assert vib.clean() == 13
    assert len(list(vib.iterimages())) == 13
    d = dict(vib.iterdisplace(inplace=False))
    for name, image in vib.iterdisplace(inplace=True):
        assert d[name] == random_dimer