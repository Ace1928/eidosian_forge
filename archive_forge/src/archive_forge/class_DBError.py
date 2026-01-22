from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBError(CausedByException):
    """Base exception for all custom database exceptions.

    :kwarg inner_exception: an original exception which was wrapped with
        DBError or its subclasses.
    """

    def __init__(self, inner_exception=None, cause=None):
        self.inner_exception = inner_exception
        super(DBError, self).__init__(str(inner_exception), cause)