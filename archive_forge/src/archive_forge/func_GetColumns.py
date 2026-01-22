import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def GetColumns(dBase, table, fieldString, user='sysdba', password='masterkey', join='', cn=None):
    """ gets a set of data from a table

      **Arguments**

       - dBase: database name

       - table: table name

       - fieldString: a string with the names of the fields to be extracted,
          this should be a comma delimited list

       - user and  password:

       - join: a join clause (omit the verb 'join')


      **Returns**

       - a list of the data

    """
    if not cn:
        cn = DbModule.connect(dBase, user, password)
    c = cn.cursor()
    cmd = 'select %s from %s' % (fieldString, table)
    if join:
        if join.strip().find('join') != 0:
            join = 'join %s' % join
        cmd += ' ' + join
    c.execute(cmd)
    return c.fetchall()