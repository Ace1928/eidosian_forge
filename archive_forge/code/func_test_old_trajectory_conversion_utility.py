import pytest
import numpy as np
import sys
from subprocess import check_call, check_output
from pathlib import Path
from ase.build import bulk
from ase.io import read, write
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io.bundletrajectory import (BundleTrajectory,
def test_old_trajectory_conversion_utility(images, trajfile):
    trajpath = Path(trajfile)
    assert trajpath.exists()
    check_call([sys.executable, '-m', 'ase.io.trajectory', trajfile])
    oldtrajpath = trajpath.with_suffix('.traj.old')
    assert oldtrajpath.exists()
    assert trajpath.exists()
    new_images = read(trajpath, ':', format='traj')
    assert_images_equal(images, new_images)