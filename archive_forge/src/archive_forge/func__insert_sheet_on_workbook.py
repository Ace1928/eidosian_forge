from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import PY3, text_type
from petl.util.base import Table, data
from petl.io.sources import read_source_from_arg, write_source_from_arg
def _insert_sheet_on_workbook(mode, sheet, wb):
    if mode == 'replace':
        try:
            ws = wb[str(sheet)]
            ws.delete_rows(1, ws.max_row)
        except KeyError:
            ws = wb.create_sheet(title=sheet)
    elif mode == 'add':
        ws = wb.create_sheet(title=sheet)
        if sheet is not None and ws.title != sheet:
            raise ValueError('Sheet %s already exists in file' % sheet)
    elif mode == 'overwrite':
        ws = wb.create_sheet(title=sheet)
    else:
        raise ValueError("Unknown mode '%s'" % mode)
    return ws