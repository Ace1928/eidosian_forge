from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_singleFile_falseCalc_multipleImages(cli, traj):
    stdout = cli.ase('diff', '--as-csv', traj)
    r = c = -1
    for rowcount, row in enumerate(stdout.split('\n')):
        for colcount, col in enumerate(row.split(',')):
            if col == 'Δx':
                r = rowcount + 2
                c = colcount
            if (rowcount == r) & (colcount == c):
                val = col
                break
    assert float(val) == 0.0