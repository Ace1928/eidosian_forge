from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class ClosedManifoldTable(ManifoldTable):
    _select = 'select name, triangulation, m, l from %s '

    def __call__(self, **kwargs):
        return ClosedManifoldTable(self._table, db_path=database_path, **kwargs)

    def _finalize(self, M, row):
        """
            Give the closed manifold a name and do the Dehn filling.
            """
        M.set_name(row[0])
        M.dehn_fill(row[2:4])