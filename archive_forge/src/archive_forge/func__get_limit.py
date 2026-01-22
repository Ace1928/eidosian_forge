import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
def _get_limit(self, session, limit_id):
    query = session.query(LimitModel).filter_by(id=limit_id)
    ref = query.first()
    if ref is None:
        raise exception.LimitNotFound(id=limit_id)
    return ref