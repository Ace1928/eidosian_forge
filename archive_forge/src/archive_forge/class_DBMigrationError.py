from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBMigrationError(DBError):
    """Wrapped migration specific exception.

    Raised when migrations couldn't be completed successfully.
    """

    def __init__(self, message):
        super(DBMigrationError, self).__init__(message)