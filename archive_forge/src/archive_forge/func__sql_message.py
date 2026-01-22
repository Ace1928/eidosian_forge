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
@_preloaded.preload_module('sqlalchemy.sql.util')
def _sql_message(self) -> str:
    util = _preloaded.sql_util
    details = [self._message()]
    if self.statement:
        stmt_detail = '[SQL: %s]' % self.statement
        details.append(stmt_detail)
        if self.params:
            if self.hide_parameters:
                details.append('[SQL parameters hidden due to hide_parameters=True]')
            else:
                params_repr = util._repr_params(self.params, 10, ismulti=self.ismulti)
                details.append('[parameters: %r]' % params_repr)
    code_str = self._code_str()
    if code_str:
        details.append(code_str)
    return '\n'.join(['(%s)' % det for det in self.detail] + details)