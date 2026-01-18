import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_write_atoms_row(db_name):
    atoms = Atoms()
    db = connect(db_name)
    uid = db.write(atoms, external_tables={'insert_tab': {'rate': 12.0, 'rate1': -10.0}, 'another_tab': {'somevalue': 1.0}})
    row = db.get(id=uid)
    row['unique_id'] = 'uniqueIDTest'
    db.write(row)