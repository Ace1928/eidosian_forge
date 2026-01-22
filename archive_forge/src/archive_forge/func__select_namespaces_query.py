from oslo_db import exception as db_exc
from oslo_db.sqlalchemy.utils import paginate_query
from oslo_log import log as logging
import sqlalchemy.exc as sa_exc
from sqlalchemy import or_
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
import glance.db.sqlalchemy.metadef_api as metadef_api
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _
def _select_namespaces_query(context, session):
    """Build the query to get all namespaces based on the context"""
    LOG.debug('context.is_admin=%(is_admin)s; context.owner=%(owner)s', {'is_admin': context.is_admin, 'owner': context.owner})
    query_ns = session.query(models.MetadefNamespace)
    if context.is_admin:
        return query_ns
    else:
        if context.owner is not None:
            query = query_ns.filter(or_(models.MetadefNamespace.owner == context.owner, models.MetadefNamespace.visibility == 'public'))
        else:
            query = query_ns.filter(models.MetadefNamespace.visibility == 'public')
        return query