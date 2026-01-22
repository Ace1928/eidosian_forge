from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
def _truncate_name(dialect, name):
    if len(name) > dialect.max_identifier_length:
        return name[0:max(dialect.max_identifier_length - 6, 0)] + '_' + hex(hash(name) % 64)[2:]
    else:
        return name