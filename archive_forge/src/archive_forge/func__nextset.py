import re
import warnings
from . import err
def _nextset(self, unbuffered=False):
    """Get the next query set."""
    conn = self._get_db()
    current_result = self._result
    if current_result is None or current_result is not conn._result:
        return None
    if not current_result.has_next:
        return None
    self._result = None
    self._clear_result()
    conn.next_result(unbuffered=unbuffered)
    self._do_get_result()
    return True