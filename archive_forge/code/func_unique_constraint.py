from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema as sa_schema
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.types import Integer
from sqlalchemy.types import NULLTYPE
from .. import util
from ..util import sqla_compat
def unique_constraint(self, name: Optional[sqla_compat._ConstraintNameDefined], source: str, local_cols: Sequence[str], schema: Optional[str]=None, **kw) -> UniqueConstraint:
    t = sa_schema.Table(source, self.metadata(), *[sa_schema.Column(n, NULLTYPE) for n in local_cols], schema=schema)
    kw['name'] = name
    uq = sa_schema.UniqueConstraint(*[t.c[n] for n in local_cols], **kw)
    t.append_constraint(uq)
    return uq