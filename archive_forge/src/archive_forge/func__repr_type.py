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
def _repr_type(type_: TypeEngine, autogen_context: AutogenContext, _skip_variants: bool=False) -> str:
    rendered = _user_defined_render('type', type_, autogen_context)
    if rendered is not False:
        return rendered
    if hasattr(autogen_context.migration_context, 'impl'):
        impl_rt = autogen_context.migration_context.impl.render_type(type_, autogen_context)
    else:
        impl_rt = None
    mod = type(type_).__module__
    imports = autogen_context.imports
    if mod.startswith('sqlalchemy.dialects'):
        match = re.match('sqlalchemy\\.dialects\\.(\\w+)', mod)
        assert match is not None
        dname = match.group(1)
        if imports is not None:
            imports.add('from sqlalchemy.dialects import %s' % dname)
        if impl_rt:
            return impl_rt
        else:
            return '%s.%r' % (dname, type_)
    elif impl_rt:
        return impl_rt
    elif not _skip_variants and sqla_compat._type_has_variants(type_):
        return _render_Variant_type(type_, autogen_context)
    elif mod.startswith('sqlalchemy.'):
        if '_render_%s_type' % type_.__visit_name__ in globals():
            fn = globals()['_render_%s_type' % type_.__visit_name__]
            return fn(type_, autogen_context)
        else:
            prefix = _sqlalchemy_autogenerate_prefix(autogen_context)
            return '%s%r' % (prefix, type_)
    else:
        prefix = _user_autogenerate_prefix(autogen_context, type_)
        return '%s%r' % (prefix, type_)