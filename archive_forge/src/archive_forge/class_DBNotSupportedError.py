from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBNotSupportedError(DBError):
    """Raised when a database backend has raised sqla.exc.NotSupportedError"""