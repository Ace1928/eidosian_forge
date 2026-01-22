from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBNonExistentDatabase(DBError):
    """Database does not exist.

    :param database: database name
    :type database: str
    """

    def __init__(self, database, inner_exception=None):
        self.database = database
        super(DBNonExistentDatabase, self).__init__(inner_exception)