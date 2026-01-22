from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class HasDescriptionCode:
    """helper which adds 'code' as an attribute and '_code_str' as a method"""
    code: Optional[str] = None

    def __init__(self, *arg: Any, **kw: Any):
        code = kw.pop('code', None)
        if code is not None:
            self.code = code
        super().__init__(*arg, **kw)
    _what_are_we = 'error'

    def _code_str(self) -> str:
        if not self.code:
            return ''
        else:
            return f'(Background on this {self._what_are_we} at: https://sqlalche.me/e/{_version_token}/{self.code})'

    def __str__(self) -> str:
        message = super().__str__()
        if self.code:
            message = '%s %s' % (message, self._code_str())
        return message