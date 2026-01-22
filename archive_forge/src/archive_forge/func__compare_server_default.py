import abc
import functools
import logging
import pprint
import re
import alembic
import alembic.autogenerate
import alembic.migration
import sqlalchemy
import sqlalchemy.exc
import sqlalchemy.sql.expression as expr
import sqlalchemy.types as types
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
@_compare_server_default.dispatch_for('mysql')
def _compare_server_default(bind, meta_col, insp_def, meta_def):
    if isinstance(meta_col.type, sqlalchemy.Boolean):
        if meta_def is None or insp_def is None:
            return meta_def != insp_def
        insp_def = insp_def.strip("'")
        return not (isinstance(meta_def.arg, expr.True_) and insp_def == '1' or (isinstance(meta_def.arg, expr.False_) and insp_def == '0'))
    if isinstance(meta_col.type, sqlalchemy.String):
        if meta_def is None or insp_def is None:
            return meta_def != insp_def
        insp_def = re.sub("^'|'$", '', insp_def)
        return meta_def.arg != insp_def