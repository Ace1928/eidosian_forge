from collections import deque
from numba.core import types, cgutils
class _Unflattener(object):
    """
    An object used to unflatten nested sequences after a given pattern
    (an arbitrarily nested sequence).
    The pattern shows the nested sequence shape desired when unflattening;
    the values it contains are irrelevant.
    """

    def __init__(self, pattern):
        self._code = self._build_unflatten_code(pattern)

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

    def unflatten(self, flatiter):
        """Rebuild a nested tuple structure.
        """
        vals = deque(flatiter)
        res = []
        cur = res
        stack = []
        for op in self._code:
            if op is _PUSH_LIST:
                stack.append(cur)
                cur.append([])
                cur = cur[-1]
            elif op is _APPEND_NEXT_VALUE:
                cur.append(vals.popleft())
            elif op is _APPEND_EMPTY_TUPLE:
                cur.append(())
            elif op is _POP:
                cur = stack.pop()
        assert not stack, stack
        assert not vals, vals
        return res