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
def _assert_all_parents_are_enabled(self, project_id):
    parents_list = self.list_project_parents(project_id)
    for project in parents_list:
        if not project.get('enabled', True):
            raise exception.ForbiddenNotSecurity(_('Cannot enable project %s since it has disabled parents') % project_id)