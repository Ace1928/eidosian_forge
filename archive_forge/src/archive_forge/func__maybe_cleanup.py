import itertools
import sys
from fixtures.callmany import (
def _maybe_cleanup(self, fn_result):
    self.addCleanup(delattr, self, 'fn_result')
    if self.cleanup_fn is not None:
        self.addCleanup(self.cleanup_fn, fn_result)
    self.fn_result = fn_result