import pydoc
from types import TracebackType
from typing import Optional, Type
from .._typing_compat import Literal
from .. import _internal
class NopPydocPager:

    def __enter__(self):
        self._orig_pager = pydoc.pager
        pydoc.pager = self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        pydoc.pager = self._orig_pager
        return False

    def __call__(self, text):
        return None