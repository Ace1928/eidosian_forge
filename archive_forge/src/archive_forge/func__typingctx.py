from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
@property
def _typingctx(self):
    return self._context.typing_context