from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def _select_worksheet(wb, worksheet, find_or_create=False):
    if worksheet is None:
        ws = wb.sheet1
    elif isinstance(worksheet, int):
        ws = wb.get_worksheet(worksheet)
    elif isinstance(worksheet, text_type):
        sheetname = text_type(worksheet)
        if find_or_create:
            if worksheet in [wbs.title for wbs in wb.worksheets()]:
                ws = wb.worksheet(sheetname)
            else:
                ws = wb.add_worksheet(sheetname, 1, 1)
        else:
            ws = wb.worksheet(sheetname)
    else:
        raise PetlArgError('Only can find worksheet by name or by number')
    return ws