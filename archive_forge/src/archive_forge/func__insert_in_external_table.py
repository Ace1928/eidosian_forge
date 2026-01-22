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
def _insert_in_external_table(self, cursor, name=None, entries=None):
    """Insert into external table"""
    if name is None or entries is None:
        return
    id = entries.pop('id')
    dtype = self._guess_type(entries)
    expected_dtype = self._get_value_type_of_table(cursor, name)
    if dtype != expected_dtype:
        raise ValueError('The provided data type for table {} is {}, while it is initialized to be of type {}'.format(name, dtype, expected_dtype))
    cursor.execute('SELECT key FROM {} WHERE id=?'.format(name), (id,))
    updates = []
    for item in cursor.fetchall():
        value = entries.pop(item[0], None)
        if value is not None:
            updates.append((value, id, self._convert_to_recognized_types(item[0])))
    sql = 'UPDATE {} SET value=? WHERE id=? AND key=?'.format(name)
    cursor.executemany(sql, updates)
    inserts = [(k, self._convert_to_recognized_types(v), id) for k, v in entries.items()]
    sql = 'INSERT INTO {} VALUES (?, ?, ?)'.format(name)
    cursor.executemany(sql, inserts)