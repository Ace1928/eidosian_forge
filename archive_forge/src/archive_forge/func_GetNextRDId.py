from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetNextRDId(conn, table, idColName='Id', leadText=''):
    """ returns the next available RDId in the database

  see RegisterItem for testing/documentation

  """
    if not leadText:
        val = conn.GetData(table=table, fields=idColName)[0][0]
        val = val.replace('_', '-')
        leadText = val.split('-')[0]
    ID = GetNextId(conn, table, idColName=idColName)
    return IndexToRDId(ID, leadText=leadText)