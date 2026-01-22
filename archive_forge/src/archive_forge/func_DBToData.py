import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def DBToData(dbName, tableName, user='sysdba', password='masterkey', dupCol=-1, what='*', where='', join='', pickleCol=-1, pickleClass=None, ensembleIds=None):
    """ constructs  an _MLData.MLDataSet_ from a database

      **Arguments**

        - dbName: the name of the database to be opened

        - tableName: the table name containing the data in the database

        - user: the user name to be used to connect to the database

        - password: the password to be used to connect to the database

        - dupCol: if nonzero specifies which column should be used to recognize
          duplicates.

      **Returns**

         an _MLData.MLDataSet_

      **Notes**

        - this uses Dbase.DataUtils functionality

    """
    from rdkit.Dbase.DbConnection import DbConnect
    conn = DbConnect(dbName, tableName, user, password)
    res = conn.GetData(fields=what, where=where, join=join, removeDups=dupCol, forceList=1)
    nPts = len(res)
    vals = [None] * nPts
    ptNames = [None] * nPts
    classWorks = True
    for i in range(nPts):
        tmp = list(res[i])
        ptNames[i] = tmp.pop(0)
        if pickleCol >= 0:
            if not pickleClass or not classWorks:
                tmp[pickleCol] = pickle.loads(str(tmp[pickleCol]))
            else:
                try:
                    tmp[pickleCol] = pickleClass(str(tmp[pickleCol]))
                except Exception:
                    tmp[pickleCol] = pickle.loads(str(tmp[pickleCol]))
                    classWorks = False
            if ensembleIds:
                tmp[pickleCol] = BitUtils.ConstructEnsembleBV(tmp[pickleCol], ensembleIds)
        elif ensembleIds:
            tmp = TakeEnsemble(tmp, ensembleIds, isDataVect=True)
        vals[i] = tmp
    varNames = conn.GetColumnNames(join=join, what=what)
    data = MLData.MLDataSet(vals, varNames=varNames, ptNames=ptNames)
    return data