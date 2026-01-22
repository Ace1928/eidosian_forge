from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBInvalidUnicodeParameter(Exception):
    """Database unicode error.

    Raised when unicode parameter is passed to a database
    without encoding directive.
    """

    def __init__(self):
        super(DBInvalidUnicodeParameter, self).__init__(_("Invalid Parameter: Encoding directive wasn't provided."))