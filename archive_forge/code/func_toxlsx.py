from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import PY3, text_type
from petl.util.base import Table, data
from petl.io.sources import read_source_from_arg, write_source_from_arg
def toxlsx(tbl, filename, sheet=None, write_header=True, mode='replace'):
    """
    Write a table to a new Excel .xlsx file.

    N.B., the sheet name is case sensitive.

    The `mode` argument controls how the file and sheet are treated:

      - `replace`: This is the default. It either replaces or adds a
        named sheet, or if no sheet name is provided, all sheets
        (overwrites the entire file).

      - `overwrite`: Always overwrites the file. This produces a file
        with a single sheet.

      - `add`: Adds a new sheet. Raises `ValueError` if a named sheet
        already exists.

    The `sheet` argument can be omitted in all cases. The new sheet
    will then get a default name.
    If the file does not exist, it will be created, unless `replace`
    mode is used with a named sheet. In the latter case, the file
    must exist and be a valid .xlsx file.
    """
    wb = _load_or_create_workbook(filename, mode, sheet)
    ws = _insert_sheet_on_workbook(mode, sheet, wb)
    if write_header:
        it = iter(tbl)
        try:
            hdr = next(it)
            flds = list(map(text_type, hdr))
            rows = itertools.chain([flds], it)
        except StopIteration:
            rows = it
    else:
        rows = data(tbl)
    for row in rows:
        ws.append(row)
    target = write_source_from_arg(filename)
    with target.open('wb') as target2:
        wb.save(target2)