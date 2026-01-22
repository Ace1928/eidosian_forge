import json
import numbers
import os
import sqlite3
import sys
from contextlib import contextmanager
import numpy as np
import ase.io.jsonio
from ase.data import atomic_numbers
from ase.calculators.calculator import all_properties
from ase.db.row import AtomsRow
from ase.db.core import (Database, ops, now, lock, invop, parse_selection,
from ase.parallel import parallel_function
def _read_external_table(self, name, id):
    """Read row from external table."""
    with self.managed_connection() as con:
        cur = con.cursor()
        cur.execute('SELECT * FROM {} WHERE id=?'.format(name), (id,))
        items = cur.fetchall()
        dictionary = dict([(item[0], item[1]) for item in items])
    return dictionary