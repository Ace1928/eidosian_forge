from contextlib import ContextDecorator, contextmanager
from django.db import (
class TransactionManagementError(ProgrammingError):
    """Transaction management is used improperly."""
    pass