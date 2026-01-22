import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_endpoint_group_in_project(self, session, endpoint_group_id, project_id):
    endpoint_group_project_ref = session.get(ProjectEndpointGroupMembership, (endpoint_group_id, project_id))
    if endpoint_group_project_ref is None:
        msg = _('Endpoint Group Project Association not found')
        raise exception.NotFound(msg)
    else:
        return endpoint_group_project_ref