import os
import os.path as op
import sys
from ase.io import read
from ase.io.formats import filetype, UnknownFileTypeError
from ase.db import connect
from ase.db.core import parse_selection
from ase.db.jsondb import JSONDatabase
from ase.db.row import atoms2dict
Check a path.

    Returns a (filetype, AtomsRow object) tuple.
    