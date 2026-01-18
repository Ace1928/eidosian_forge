import pytest
import numpy as np
from ase.parallel import world
from ase.build import molecule, fcc111
from ase.build.attach import (attach, attach_randomly,
def test_pbc():
    """Attach two molecules and check attachment considers pbc"""
    m1 = molecule('C6H6')
    m1.cell = (20, 1, 1)
    m1.translate((16, 0, 0))
    m1.pbc = (1, 0, 0)
    m2 = molecule('NH3')
    distance = 2.0
    m12 = attach(m1, m2, distance)
    for atom in m12[-4:]:
        assert atom.position[0] < 2