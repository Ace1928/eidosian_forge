import sys
from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetTableNames(dBase, user='sysdba', password='masterkey', includeViews=0, cn=None):
    """ returns a list of tables available in a database

      **Arguments**

        - dBase: the name of the DB file to be used

        - user: the username for DB access

        - password: the password to be used for DB access

        - includeViews: if this is non-null, the views in the db will
          also be returned

      **Returns**

        - a list of table names (strings)

    """
    if not cn:
        try:
            cn = DbModule.connect(dBase, user, password)
        except Exception:
            print('Problems opening database: %s' % dBase)
            return []
    c = cn.cursor()
    if not includeViews:
        comm = DbModule.getTablesSql
    else:
        comm = DbModule.getTablesAndViewsSql
    c.execute(comm)
    names = [str(x[0]).upper() for x in c.fetchall()]
    if RDConfig.usePgSQL and 'PG_LOGDIR_LS' in names:
        names.remove('PG_LOGDIR_LS')
    return names