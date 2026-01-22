from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class NonorientableClosedCensus(ClosedManifoldTable):
    """
        Iterator for 17 nonorientable closed hyperbolic manifolds from the
        census by Hodgson and Weeks.

        >>> for M in NonorientableClosedCensus[:3]: print(M, M.volume()) # doctest: +NUMERIC6
        ... 
        m018(1,0) 2.02988321
        m177(1,0) 2.56897060
        m153(1,0) 2.66674478
        """

    def __init__(self, **kwargs):
        return ClosedManifoldTable.__init__(self, table='nonorientable_closed_view', db_path=database_path, **kwargs)