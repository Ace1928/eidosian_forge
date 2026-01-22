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
@event.listens_for(orm.session.Session, 'before_commit')
def _load_one_to_manys(session):
    if session.new:
        session.flush()
    if session.get_transaction().nested:
        return
    for new_object in session.info.pop('_load_rels', []):
        if new_object not in session:
            continue
        state = sqlalchemy.inspect(new_object)
        session.enable_relationship_loading(new_object)
        for relationship_attr in state.mapper.relationships:
            if relationship_attr.lazy not in ('joined', 'subquery'):
                continue
            if relationship_attr.key not in state.dict:
                getattr(new_object, relationship_attr.key)
                if relationship_attr.key not in state.dict:
                    msg = 'Relationship %s attributes must be loaded in db object %s' % (relationship_attr.key, state.dict)
                    raise AssertionError(msg)