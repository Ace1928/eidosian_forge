from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def _data2rows(self, raw_data):
    """Return list of Row,
        the raw data as rows of cells.
        """
    _Cell = self._Cell
    _Row = self._Row
    rows = []
    for datarow in raw_data:
        dtypes = cycle(self._datatypes)
        newrow = _Row(datarow, datatype='data', table=self, celltype=_Cell)
        for cell in newrow:
            cell.datatype = next(dtypes)
            cell.row = newrow
        rows.append(newrow)
    return rows