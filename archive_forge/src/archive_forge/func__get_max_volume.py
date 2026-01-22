from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def _get_max_volume(self):
    where_clause = 'where ' + self._filter if self._filter else ''
    vol_query = 'select max(volume) from %s %s' % (self._table, where_clause)
    cursor = self._cursor.execute(vol_query)
    self._max_volume = cursor.fetchone()[0]