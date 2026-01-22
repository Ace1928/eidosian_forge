from __future__ import annotations
from . import exc
from . import util as orm_util
from .base import PassiveFlag
def _raise_col_to_prop(isdest, source_mapper, source_column, dest_mapper, dest_column, err):
    if isdest:
        raise exc.UnmappedColumnError("Can't execute sync rule for destination column '%s'; mapper '%s' does not map this column.  Try using an explicit `foreign_keys` collection which does not include this column (or use a viewonly=True relation)." % (dest_column, dest_mapper)) from err
    else:
        raise exc.UnmappedColumnError("Can't execute sync rule for source column '%s'; mapper '%s' does not map this column.  Try using an explicit `foreign_keys` collection which does not include destination column '%s' (or use a viewonly=True relation)." % (source_column, source_mapper, dest_column)) from err