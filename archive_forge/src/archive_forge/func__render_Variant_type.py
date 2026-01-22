from __future__ import annotations
from io import StringIO
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from mako.pygen import PythonPrinter
from sqlalchemy import schema as sa_schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.elements import conv
from sqlalchemy.sql.elements import quoted_name
from .. import util
from ..operations import ops
from ..util import sqla_compat
def _render_Variant_type(type_: TypeEngine, autogen_context: AutogenContext) -> str:
    base_type, variant_mapping = sqla_compat._get_variant_mapping(type_)
    base = _repr_type(base_type, autogen_context, _skip_variants=True)
    assert base is not None and base is not False
    for dialect in sorted(variant_mapping):
        typ = variant_mapping[dialect]
        base += '.with_variant(%s, %r)' % (_repr_type(typ, autogen_context, _skip_variants=True), dialect)
    return base