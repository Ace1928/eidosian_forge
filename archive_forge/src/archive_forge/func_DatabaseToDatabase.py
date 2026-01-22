import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def DatabaseToDatabase(fromDb, fromTbl, toDb, toTbl, fields='*', join='', where='', user='sysdba', password='masterkey', keyCol=None, nullMarker='None'):
    """

     FIX: at the moment this is a hack

    """
    sio = StringIO()
    sio.write(DatabaseToText(fromDb, fromTbl, fields=fields, join=join, where=where, user=user, password=password))
    sio.seek(0)
    TextFileToDatabase(toDb, toTbl, sio, user=user, password=password, keyCol=keyCol, nullMarker=nullMarker)