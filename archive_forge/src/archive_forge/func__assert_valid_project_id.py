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
def _assert_valid_project_id(self, project_id):
    if project_id is None:
        msg = _('Project field is required and cannot be empty.')
        raise exception.ValidationError(message=msg)
    self.get_project(project_id)