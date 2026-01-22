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
def _get_column_info(self, name, type_, nullable, default, primary_key, generated, persisted, tablesql):
    if generated:
        type_ = re.sub('generated', '', type_, flags=re.IGNORECASE)
        type_ = re.sub('always', '', type_, flags=re.IGNORECASE).strip()
    coltype = self._resolve_type_affinity(type_)
    if default is not None:
        default = str(default)
    colspec = {'name': name, 'type': coltype, 'nullable': nullable, 'default': default, 'primary_key': primary_key}
    if generated:
        sqltext = ''
        if tablesql:
            pattern = '[^,]*\\s+AS\\s+\\(([^,]*)\\)\\s*(?:virtual|stored)?'
            match = re.search(re.escape(name) + pattern, tablesql, re.IGNORECASE)
            if match:
                sqltext = match.group(1)
        colspec['computed'] = {'sqltext': sqltext, 'persisted': persisted}
    return colspec