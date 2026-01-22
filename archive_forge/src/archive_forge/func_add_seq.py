from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
def add_seq(c, tbl):
    c._init_items(schema.Sequence(_truncate_name(config.db.dialect, tbl.name + '_' + c.name + '_seq'), optional=True))