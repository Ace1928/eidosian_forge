from __future__ import division, print_function, absolute_import
import locale
from petl.compat import izip_longest, next, xrange, BytesIO
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
class XLSView(Table):

    def __init__(self, filename, sheet=None, use_view=True, **kwargs):
        self.filename = filename
        self.sheet = sheet
        self.use_view = use_view
        self.kwargs = kwargs

    def __iter__(self):
        if self.use_view:
            from petl.io import xlutils_view
            source = read_source_from_arg(self.filename)
            with source.open('rb') as source2:
                source3 = source2.read()
                wb = xlutils_view.View(source3, **self.kwargs)
                if self.sheet is None:
                    ws = wb[0]
                else:
                    ws = wb[self.sheet]
                for row in ws:
                    yield tuple(row)
        else:
            import xlrd
            source = read_source_from_arg(self.filename)
            with source.open('rb') as source2:
                source3 = source2.read()
                with xlrd.open_workbook(file_contents=source3, on_demand=True, **self.kwargs) as wb:
                    if self.sheet is None:
                        ws = wb.sheet_by_index(0)
                    elif isinstance(self.sheet, int):
                        ws = wb.sheet_by_index(self.sheet)
                    else:
                        ws = wb.sheet_by_name(str(self.sheet))
                    for rownum in xrange(ws.nrows):
                        yield tuple(ws.row_values(rownum))