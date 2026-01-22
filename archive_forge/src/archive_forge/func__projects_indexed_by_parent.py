from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _projects_indexed_by_parent(projects_list):
    projects_by_parent = {}
    for proj in projects_list:
        parent_id = proj.get('parent_id')
        if parent_id:
            if parent_id in projects_by_parent:
                projects_by_parent[parent_id].append(proj)
            else:
                projects_by_parent[parent_id] = [proj]
    return projects_by_parent