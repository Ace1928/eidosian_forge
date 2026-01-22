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
def _guess_type(self, entries):
    """Guess the type based on the first entry."""
    values = [v for _, v in entries.items()]
    all_types = [type(v) for v in values]
    if any([t != all_types[0] for t in all_types]):
        typenames = [t.__name__ for t in all_types]
        raise ValueError('Inconsistent datatypes in the table. given types: {}'.format(typenames))
    val = values[0]
    if isinstance(val, int) or np.issubdtype(type(val), np.integer):
        return 'INTEGER'
    if isinstance(val, float) or np.issubdtype(type(val), np.floating):
        return 'REAL'
    if isinstance(val, str):
        return 'TEXT'
    raise ValueError('Unknown datatype!')