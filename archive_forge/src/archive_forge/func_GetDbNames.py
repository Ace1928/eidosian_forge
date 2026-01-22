import sys
from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetDbNames(user='sysdba', password='masterkey', dirName='.', dBase='::template1', cn=None):
    """ returns a list of databases that are available

      **Arguments**

        - user: the username for DB access

        - password: the password to be used for DB access

      **Returns**

        - a list of db names (strings)

    """
    if DbModule.getDbSql:
        if not cn:
            try:
                cn = DbModule.connect(dBase, user, password)
            except Exception:
                print('Problems opening database: %s' % dBase)
                return []
        c = cn.cursor()
        c.execute(DbModule.getDbSql)
        if RDConfig.usePgSQL:
            names = ['::' + str(x[0]) for x in c.fetchall()]
        else:
            names = ['::' + str(x[0]) for x in c.fetchall()]
        names.remove(dBase)
    elif DbModule.fileWildcard:
        import glob
        import os.path
        names = glob.glob(os.path.join(dirName, DbModule.fileWildcard))
    else:
        names = []
    return names