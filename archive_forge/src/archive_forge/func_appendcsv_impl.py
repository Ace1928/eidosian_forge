import io
import csv
import logging
from petl.util.base import Table, data
def appendcsv_impl(table, source, **kwargs):
    _writecsv(table, source=source, mode='ab', **kwargs)