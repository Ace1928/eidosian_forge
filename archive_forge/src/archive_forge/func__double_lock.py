from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
def _double_lock(self, lock_source, lock_target):
    """Take out two locks, rolling back the first if the second throws."""
    lock_source()
    try:
        lock_target()
    except Exception:
        self.source.unlock()
        raise