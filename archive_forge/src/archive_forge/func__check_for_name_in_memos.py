from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from ... import schema
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import roles
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import InvokeCreateDDLBase
from ...sql.ddl import InvokeDropDDLBase
def _check_for_name_in_memos(self, checkfirst, kw):
    """Look in the 'ddl runner' for 'memos', then
        note our name in that collection.

        This to ensure a particular named type is operated
        upon only once within any kind of create/drop
        sequence without relying upon "checkfirst".

        """
    if not self.create_type:
        return True
    if '_ddl_runner' in kw:
        ddl_runner = kw['_ddl_runner']
        type_name = f'pg_{self.__visit_name__}'
        if type_name in ddl_runner.memo:
            existing = ddl_runner.memo[type_name]
        else:
            existing = ddl_runner.memo[type_name] = set()
        present = (self.schema, self.name) in existing
        existing.add((self.schema, self.name))
        return present
    else:
        return False