import contextlib
import copy
import functools
import weakref
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import excutils
from osprofiler import opts as profiler_opts
import osprofiler.sqlalchemy
from pecan import util as p_util
import sqlalchemy
from sqlalchemy import event  # noqa
from sqlalchemy import exc as sql_exc
from sqlalchemy import orm
from sqlalchemy.orm import exc
from neutron_lib._i18n import _
from neutron_lib.db import model_base
from neutron_lib import exceptions
from neutron_lib.objects import exceptions as obj_exc
def _expire_for_fk_change(target, fk_value, relationship_prop, column_attr):
    """Expire relationship attributes when a many-to-one column changes."""
    sess = orm.object_session(target)
    if sess is not None:
        if relationship_prop.back_populates and relationship_prop.key in target.__dict__:
            obj = getattr(target, relationship_prop.key)
            if obj is not None and sqlalchemy.inspect(obj).persistent:
                sess.expire(obj, [relationship_prop.back_populates])
        if sqlalchemy.inspect(target).persistent:
            sess.expire(target, [relationship_prop.key])
        if relationship_prop.back_populates:
            target.__dict__[column_attr] = fk_value
            new = getattr(target, relationship_prop.key)
            if new is not None:
                if sqlalchemy.inspect(new).persistent:
                    sess.expire(new, [relationship_prop.back_populates])
    else:
        if target not in _emit_on_pending:
            _emit_on_pending[target] = []
        _emit_on_pending[target].append((fk_value, relationship_prop, column_attr))