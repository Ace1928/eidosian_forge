import numpy as np
import pytest
from ase.optimize import FIRE, BFGS
from ase.data import s22
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io import Trajectory
import ase.units as units
@pytest.mark.parametrize('cls', [FIRE, BFGS])
def test_optimization_log_and_trajectory_length(cls, testdir):
    logfile = 'opt.log'
    trajectory = 'opt.traj'
    atoms = make_dimer()
    print('Testing', str(cls))
    with cls(atoms, logfile=logfile, trajectory=trajectory) as opt:
        opt.run(0.2)
        opt.run(0.1)
    with open(logfile, 'rt') as lf:
        lines = [l for l in lf]
    loglines = len(lines)
    print('Number of lines in log file:', loglines)
    with Trajectory(trajectory) as traj:
        trajframes = len(traj)
    print('Number of frames in trajectory:', trajframes)
    assert loglines == trajframes + 1