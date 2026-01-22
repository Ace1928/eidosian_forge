from rdkit.Dbase import DbInfo, DbModule, DbUtils
def InsertData(self, tableName, vals):
    """ inserts data into a table

    **Arguments**

      - tableName: the name of the table to manipulate

      - vals: a sequence with the values to be inserted

    """
    c = self.GetCursor()
    if type(vals) != tuple:
        vals = tuple(vals)
    insTxt = '(' + ','.join([DbModule.placeHolder] * len(vals)) + ')'
    cmd = 'insert into %s values %s' % (tableName, insTxt)
    try:
        c.execute(cmd, vals)
    except Exception:
        import traceback
        print('insert failed:')
        print(cmd)
        print('the error was:')
        traceback.print_exc()
        raise DbError('Insert Failed')