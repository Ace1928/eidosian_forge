import os
import pytest
from ase.db import connect
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import molecule
def test_write_read(db):
    co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.1)])
    uid = db.write(co, tag=1, type='molecule')
    co_db = db.get(id=uid)
    atoms_db = co_db.toatoms()
    assert len(atoms_db) == 2
    assert atoms_db[0].symbol == co[0].symbol
    assert atoms_db[1].symbol == co[1].symbol
    assert co_db.tag == 1
    assert co_db.type == 'molecule'