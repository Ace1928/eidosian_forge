from rdkit.Dbase import DbInfo, DbModule, DbUtils
def KillCursor(self):
    """ closes the cursor

    """
    self.cursor = None
    if self.cn is not None:
        self.cn.close()
    self.cn = None