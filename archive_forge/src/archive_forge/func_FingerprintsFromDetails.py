import getopt
import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Murtagh
def FingerprintsFromDetails(details, reportFreq=10):
    data = None
    if details.dbName and details.tableName:
        from rdkit.Dbase import DbInfo
        from rdkit.Dbase.DbConnection import DbConnect
        from rdkit.ML.Data import DataUtils
        try:
            conn = DbConnect(details.dbName, details.tableName)
        except Exception:
            import traceback
            error(f'Problems establishing connection to database: {details.dbName}|{details.tableName}\n')
            traceback.print_exc()
        if not details.idName:
            details.idName = DbInfo.GetColumnNames(details.dbName, details.tableName)[0]
        dataSet = DataUtils.DBToData(details.dbName, details.tableName, what=f'{details.idName},{details.smilesName}')
        idCol = 0
        smiCol = 1
    elif details.inFileName and details.useSmiles:
        from rdkit.ML.Data import DataUtils
        conn = None
        if not details.idName:
            details.idName = 'ID'
        try:
            dataSet = DataUtils.TextFileToData(details.inFileName, onlyCols=[details.idName, details.smilesName])
        except IOError:
            import traceback
            error(f'Problems reading from file {details.inFileName}\n')
            traceback.print_exc()
        idCol = 0
        smiCol = 1
    elif details.inFileName and details.useSD:
        conn = None
        if not details.idName:
            details.idName = 'ID'
        dataSet = []
        try:
            s = Chem.SDMolSupplier(details.inFileName)
        except Exception:
            import traceback
            error(f'Problems reading from file {details.inFileName}\n')
            traceback.print_exc()
        else:
            while 1:
                try:
                    m = s.next()
                except StopIteration:
                    break
                if m:
                    dataSet.append(m)
                    if reportFreq > 0 and (not len(dataSet) % reportFreq):
                        message(f'Read {len(dataSet)} molecules\n')
                        if 0 < details.maxMols <= len(dataSet):
                            break
        for i, mol in enumerate(dataSet):
            if mol.HasProp(details.idName):
                nm = mol.GetProp(details.idName)
            else:
                nm = mol.GetProp('_Name')
            dataSet[i] = (nm, mol)
    else:
        dataSet = None
    fps = None
    if dataSet and (not details.useSD):
        data = dataSet.GetNamedData()
        if not details.molPklName:
            fps = FingerprintsFromSmiles(data, idCol, smiCol, **details.__dict__)
        else:
            fps = FingerprintsFromPickles(data, idCol, smiCol, **details.__dict__)
    elif dataSet and details.useSD:
        fps = FingerprintsFromMols(dataSet, **details.__dict__)
    if fps:
        if details.outFileName:
            outF = open(details.outFileName, 'wb+')
            for i in range(len(fps)):
                pickle.dump(fps[i], outF)
            outF.close()
        dbName = details.outDbName or details.dbName
        if details.outTableName and dbName:
            from rdkit.Dbase import DbModule, DbUtils
            from rdkit.Dbase.DbConnection import DbConnect
            conn = DbConnect(dbName)
            colTypes = DbUtils.TypeFinder(data, len(data), len(data[0]))
            typeStrs = DbUtils.GetTypeStrings([details.idName, details.smilesName], colTypes, keyCol=details.idName)
            cols = f'{typeStrs[0]}, {details.fpColName} {DbModule.binaryTypeName}'
            if details.replaceTable or details.outTableName.upper() not in [x.upper() for x in conn.GetTableNames()]:
                conn.AddTable(details.outTableName, cols)
            for ID, fp in fps:
                tpl = (ID, DbModule.binaryHolder(fp.ToBinary()))
                conn.InsertData(details.outTableName, tpl)
            conn.Commit()
    return fps