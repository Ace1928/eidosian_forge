import os
import pytest
from ase.db import connect
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import molecule
def test_write_read_with_calculator(db, h2o):
    calc = EMT(dummy_param=2.4)
    h2o.calc = calc
    uid = db.write(h2o)
    h2o_db = db.get(id=uid).toatoms(attach_calculator=True)
    calc_db = h2o_db.calc
    assert calc_db.parameters['dummy_param'] == 2.4
    db.get_atoms(H=2)