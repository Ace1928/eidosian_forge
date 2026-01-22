import pydoc
from types import TracebackType
from typing import Optional, Type
from .._typing_compat import Literal
from .. import _internal
class _Helper(_internal._Helper):

    def __init__(self, repl=None):
        self._repl = repl
        pydoc.pager = self.pager
        super().__init__()

    def pager(self, output):
        self._repl.pager(output)

    def __call__(self, *args, **kwargs):
        if self._repl.reevaluating:
            with NopPydocPager():
                return super().__call__(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)