from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def appendtsv(table, source=None, encoding=None, errors='strict', write_header=False, **csvargs):
    """
    Convenience function, as :func:`petl.io.csv.appendcsv` but with different
    default dialect (tab delimited).

    """
    csvargs.setdefault('dialect', 'excel-tab')
    return appendcsv(table, source=source, encoding=encoding, errors=errors, write_header=write_header, **csvargs)