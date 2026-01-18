from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
def visit_on_duplicate_key_update(self, on_duplicate, **kw):
    statement = self.current_executable
    if on_duplicate._parameter_ordering:
        parameter_ordering = [coercions.expect(roles.DMLColumnRole, key) for key in on_duplicate._parameter_ordering]
        ordered_keys = set(parameter_ordering)
        cols = [statement.table.c[key] for key in parameter_ordering if key in statement.table.c] + [c for c in statement.table.c if c.key not in ordered_keys]
    else:
        cols = statement.table.c
    clauses = []
    requires_mysql8_alias = self.dialect._requires_alias_for_on_duplicate_key
    if requires_mysql8_alias:
        if statement.table.name.lower() == 'new':
            _on_dup_alias_name = 'new_1'
        else:
            _on_dup_alias_name = 'new'
    for column in (col for col in cols if col.key in on_duplicate.update):
        val = on_duplicate.update[column.key]
        if coercions._is_literal(val):
            val = elements.BindParameter(None, val, type_=column.type)
            value_text = self.process(val.self_group(), use_schema=False)
        else:

            def replace(obj):
                if isinstance(obj, elements.BindParameter) and obj.type._isnull:
                    obj = obj._clone()
                    obj.type = column.type
                    return obj
                elif isinstance(obj, elements.ColumnClause) and obj.table is on_duplicate.inserted_alias:
                    if requires_mysql8_alias:
                        column_literal_clause = f'{_on_dup_alias_name}.{self.preparer.quote(obj.name)}'
                    else:
                        column_literal_clause = f'VALUES({self.preparer.quote(obj.name)})'
                    return literal_column(column_literal_clause)
                else:
                    return None
            val = visitors.replacement_traverse(val, {}, replace)
            value_text = self.process(val.self_group(), use_schema=False)
        name_text = self.preparer.quote(column.name)
        clauses.append('%s = %s' % (name_text, value_text))
    non_matching = set(on_duplicate.update) - {c.key for c in cols}
    if non_matching:
        util.warn("Additional column names not matching any column keys in table '%s': %s" % (self.statement.table.name, ', '.join(("'%s'" % c for c in non_matching))))
    if requires_mysql8_alias:
        return f'AS {_on_dup_alias_name} ON DUPLICATE KEY UPDATE {', '.join(clauses)}'
    else:
        return f'ON DUPLICATE KEY UPDATE {', '.join(clauses)}'