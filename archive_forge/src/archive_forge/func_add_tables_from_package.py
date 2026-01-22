from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def add_tables_from_package(package_name, must_succeed=True):
    """
    Given a string with the name of an importable Python package that
    implements a "get_tables" function, load all the tables provided
    and put the results where the rest of SnapPy can find them.
    Returns an ordered dictionary of pairs (table_name, table).
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        if not must_succeed:
            return dict()
        else:
            raise ImportError('ManifoldTable package %s not found' % package_name)
    new_tables = collections.OrderedDict()
    for table in package.get_tables(ManifoldTable):
        name = table.__class__.__name__
        new_tables[name] = table
        __all_tables__[name] = table
        setattr(this_module, name, table)
    if not hasattr(this_module, '__test__'):
        this_module.__test__ = dict()
    for name, table in new_tables.items():
        this_module.__test__[name] = table.__class__
    return new_tables