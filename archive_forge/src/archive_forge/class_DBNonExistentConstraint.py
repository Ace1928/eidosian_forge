from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBNonExistentConstraint(DBError):
    """Constraint does not exist.

    :param table: table name
    :type table: str
    :param constraint: constraint name
    :type table: str
    """

    def __init__(self, table, constraint, inner_exception=None):
        self.table = table
        self.constraint = constraint
        super(DBNonExistentConstraint, self).__init__(inner_exception)