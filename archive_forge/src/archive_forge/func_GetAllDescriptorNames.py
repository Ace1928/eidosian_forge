from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def GetAllDescriptorNames(db, tbl1, tbl2, user='sysdba', password='masterkey'):
    """ gets possible descriptor names from a database

    **Arguments**

      - db: the name of the database to use

      - tbl1: the name of the table to be used for reading descriptor values

      - tbl2: the name of the table to be used for reading notes about the
        descriptors (*descriptions of the descriptors if you like*)

      - user: the user name for DB access

      - password: the password for DB access

    **Returns**

      a 2-tuple containing:

        1) a list of column names

        2) a list of column descriptors

    **Notes**

      - this uses _Dbase.DbInfo_  and Dfunctionality for querying the database

      - it is assumed that tbl2 includes 'property' and 'notes' columns

  """
    from rdkit.Dbase.DbConnection import DbConnect
    conn = DbConnect(db, user=user, password=password)
    colNames = conn.GetColumnNames(table=tbl1)
    colDesc = map(lambda x: (x[0].upper(), x[1]), conn.GetColumns('property,notes', table=tbl2))
    for name, desc in countOptions:
        colNames.append(name)
        colDesc.append((name, desc))
    return (colNames, colDesc)