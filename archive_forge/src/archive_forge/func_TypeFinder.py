import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def TypeFinder(data, nRows, nCols, nullMarker=None):
    """

      finds the types of the columns in _data_

      if nullMarker is not None, elements of the data table which are
        equal to nullMarker will not count towards setting the type of
        their columns.

    """
    priorities = {float: 3, int: 2, str: 1, -1: -1}
    res = [None] * nCols
    for col in range(nCols):
        typeHere = [-1, 1]
        for row in range(nRows):
            d = data[row][col]
            if d is None:
                continue
            locType = type(d)
            if locType != float and locType != int:
                locType = str
                try:
                    d = str(d)
                except UnicodeError as msg:
                    print('cannot convert text from row %d col %d to a string' % (row + 2, col))
                    print('\t>%s' % repr(d))
                    raise UnicodeError(msg)
            else:
                typeHere[1] = max(typeHere[1], len(str(d)))
            if isinstance(d, str):
                if nullMarker is None or d != nullMarker:
                    l = max(len(d), typeHere[1])
                    typeHere = [str, l]
            else:
                try:
                    fD = float(int(d))
                except OverflowError:
                    locType = float
                else:
                    if fD == d:
                        locType = int
                if not isinstance(typeHere[0], str) and priorities[locType] > priorities[typeHere[0]]:
                    typeHere[0] = locType
        res[col] = typeHere
    return res