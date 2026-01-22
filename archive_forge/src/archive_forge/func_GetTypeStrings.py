import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def GetTypeStrings(colHeadings, colTypes, keyCol=None):
    """  returns a list of SQL type strings
    """
    typeStrs = []
    for i in range(len(colTypes)):
        typ = colTypes[i]
        if typ[0] == float:
            typeStrs.append('%s double precision' % colHeadings[i])
        elif typ[0] == int:
            typeStrs.append('%s integer' % colHeadings[i])
        else:
            typeStrs.append('%s varchar(%d)' % (colHeadings[i], typ[1]))
        if colHeadings[i] == keyCol:
            typeStrs[-1] = '%s not null primary key' % typeStrs[-1]
    return typeStrs