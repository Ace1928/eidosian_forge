from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
def visit_on_conflict_do_update(self, on_conflict, **kw):
    clause = on_conflict
    target_text = self._on_conflict_target(on_conflict, **kw)
    action_set_ops = []
    set_parameters = dict(clause.update_values_to_set)
    insert_statement = self.stack[-1]['selectable']
    cols = insert_statement.table.c
    for c in cols:
        col_key = c.key
        if col_key in set_parameters:
            value = set_parameters.pop(col_key)
        elif c in set_parameters:
            value = set_parameters.pop(c)
        else:
            continue
        if coercions._is_literal(value):
            value = elements.BindParameter(None, value, type_=c.type)
        elif isinstance(value, elements.BindParameter) and value.type._isnull:
            value = value._clone()
            value.type = c.type
        value_text = self.process(value.self_group(), use_schema=False)
        key_text = self.preparer.quote(c.name)
        action_set_ops.append('%s = %s' % (key_text, value_text))
    if set_parameters:
        util.warn("Additional column names not matching any column keys in table '%s': %s" % (self.current_executable.table.name, ', '.join(("'%s'" % c for c in set_parameters))))
        for k, v in set_parameters.items():
            key_text = self.preparer.quote(k) if isinstance(k, str) else self.process(k, use_schema=False)
            value_text = self.process(coercions.expect(roles.ExpressionElementRole, v), use_schema=False)
            action_set_ops.append('%s = %s' % (key_text, value_text))
    action_text = ', '.join(action_set_ops)
    if clause.update_whereclause is not None:
        action_text += ' WHERE %s' % self.process(clause.update_whereclause, include_table=True, use_schema=False)
    return 'ON CONFLICT %s DO UPDATE SET %s' % (target_text, action_text)