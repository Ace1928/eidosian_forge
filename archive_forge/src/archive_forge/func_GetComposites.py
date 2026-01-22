from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import BuildComposite, CompositeRun, ScreenComposite
from rdkit.ML.Composite import AdjustComposite
from rdkit.ML.Data import DataUtils, SplitData
def GetComposites(details):
    res = []
    if details.persistTblName and details.inNote:
        conn = DbConnect(details.dbName, details.persistTblName)
        mdls = conn.GetData(fields='MODEL', where="where note='%s'" % details.inNote)
        for row in mdls:
            rawD = row[0]
            res.append(pickle.loads(str(rawD)))
    elif details.composFileName:
        res.append(pickle.load(open(details.composFileName, 'rb')))
    return res