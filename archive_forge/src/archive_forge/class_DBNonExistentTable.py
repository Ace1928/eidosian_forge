from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBNonExistentTable(DBError):
    """Table does not exist.

    :param table: table name
    :type table: str
    """

    def __init__(self, table, inner_exception=None):
        self.table = table
        super(DBNonExistentTable, self).__init__(inner_exception)