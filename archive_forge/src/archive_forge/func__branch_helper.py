import contextlib
import functools
from llvmlite.ir import instructions, types, values
@contextlib.contextmanager
def _branch_helper(self, bbenter, bbexit):
    self.position_at_end(bbenter)
    yield bbexit
    if self.basic_block.terminator is None:
        self.branch(bbexit)