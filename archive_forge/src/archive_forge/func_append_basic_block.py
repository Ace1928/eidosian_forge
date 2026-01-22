import contextlib
import functools
from llvmlite.ir import instructions, types, values
def append_basic_block(self, name=''):
    """
        Append a basic block, with the given optional *name*, to the current
        function.  The current block is not changed.  The new block is returned.
        """
    return self.function.append_basic_block(name)