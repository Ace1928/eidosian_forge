from fontTools.ttLib.tables import otTables as ot
from .table_builder import TableUnbuilder
def unbuildPaint(self, paint):
    assert isinstance(paint, ot.Paint)
    return self.tableUnbuilder.unbuild(paint)