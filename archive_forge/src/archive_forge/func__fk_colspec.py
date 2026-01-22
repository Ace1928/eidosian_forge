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
def _fk_colspec(fk: ForeignKey, metadata_schema: Optional[str], namespace_metadata: MetaData) -> str:
    """Implement a 'safe' version of ForeignKey._get_colspec() that
    won't fail if the remote table can't be resolved.

    """
    colspec = fk._get_colspec()
    tokens = colspec.split('.')
    tname, colname = tokens[-2:]
    if metadata_schema is not None and len(tokens) == 2:
        table_fullname = '%s.%s' % (metadata_schema, tname)
    else:
        table_fullname = '.'.join(tokens[0:-1])
    if not fk.link_to_name and fk.parent is not None and (fk.parent.table is not None):
        if table_fullname in namespace_metadata.tables:
            col = namespace_metadata.tables[table_fullname].c.get(colname)
            if col is not None:
                colname = _ident(col.name)
    colspec = '%s.%s' % (table_fullname, colname)
    return colspec