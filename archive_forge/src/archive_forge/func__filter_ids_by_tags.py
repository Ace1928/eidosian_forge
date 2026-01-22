from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def _filter_ids_by_tags(self, query, tags):
    filtered_ids = []
    subq = query.filter(sql_model.ProjectTag.name.in_(tags))
    for ptag in subq:
        subq_tags = query.filter(sql_model.ProjectTag.project_id == ptag['project_id'])
        result = map(lambda x: x['name'], subq_tags.all())
        if set(tags) <= set(result):
            filtered_ids.append(ptag['project_id'])
    return filtered_ids