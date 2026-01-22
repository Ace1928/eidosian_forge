import concurrent.futures._base
import logging
import reprlib
import sys
import traceback
from . import events
def _copy_state(self, other):
    """Internal helper to copy state from another Future.

        The other Future may be a concurrent.futures.Future.
        """
    assert other.done()
    if self.cancelled():
        return
    assert not self.done()
    if other.cancelled():
        self.cancel()
    else:
        exception = other.exception()
        if exception is not None:
            self.set_exception(exception)
        else:
            result = other.result()
            self.set_result(result)