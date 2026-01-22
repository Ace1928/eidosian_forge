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
def _get_all_by_resource_types(context, session, filters, marker=None, limit=None, sort_key=None, sort_dir=None):
    """get all visible namespaces for the specified resource_types"""
    resource_types = filters['resource_types']
    resource_type_list = resource_types.split(',')
    db_recs = session.query(models.MetadefResourceType).join(models.MetadefResourceType.associations).filter(models.MetadefResourceType.name.in_(resource_type_list)).with_entities(models.MetadefResourceType.name, models.MetadefNamespaceResourceType.namespace_id)
    namespace_id_list = []
    for name, namespace_id in db_recs:
        namespace_id_list.append(namespace_id)
    if len(namespace_id_list) == 0:
        return []
    filters2 = filters
    filters2.update({'id_list': namespace_id_list})
    return _get_all(context, session, filters2, marker, limit, sort_key, sort_dir)