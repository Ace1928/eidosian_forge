import collections
import enum
from . import exceptions
from . import mixins
from . import tasks
def _wake_up_first(self):
    """Wake up the first waiter if it isn't done."""
    if not self._waiters:
        return
    try:
        fut = next(iter(self._waiters))
    except StopIteration:
        return
    if not fut.done():
        fut.set_result(True)