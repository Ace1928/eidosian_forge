from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def _get_children(self, session, project_ids, domain_id=None):
    query = session.query(sql_model.Project)
    query = query.filter(sql_model.Project.parent_id.in_(project_ids))
    project_refs = query.all()
    return [project_ref.to_dict() for project_ref in project_refs]