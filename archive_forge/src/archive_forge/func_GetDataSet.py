from warnings import warn
from rdkit import RDConfig
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def GetDataSet(self, **kwargs):
    """ Returns a MLDataSet pulled from a database using our stored
    values.

    """
    from rdkit.ML.Data import DataUtils
    data = DataUtils.DBToData(self.dbName, self.tableName, user=self.dbUser, password=self.dbPassword, what=self.dbWhat, where=self.dbWhere, join=self.dbJoin, **kwargs)
    return data