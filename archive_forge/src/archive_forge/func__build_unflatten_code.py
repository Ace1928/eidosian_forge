from collections import deque
from numba.core import types, cgutils
def _build_unflatten_code(self, iterable):
    """Build the unflatten opcode sequence for the given *iterable* structure
        (an iterable of nested sequences).
        """
    code = []

    def rec(iterable):
        for i in iterable:
            if isinstance(i, (tuple, list)):
                if len(i) > 0:
                    code.append(_PUSH_LIST)
                    rec(i)
                    code.append(_POP)
                else:
                    code.append(_APPEND_EMPTY_TUPLE)
            else:
                code.append(_APPEND_NEXT_VALUE)
    rec(iterable)
    return code