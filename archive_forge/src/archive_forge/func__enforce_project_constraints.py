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
def _enforce_project_constraints(self, project_ref):
    if project_ref.get('is_domain'):
        self._assert_is_domain_project_constraints(project_ref)
    else:
        self._assert_regular_project_constraints(project_ref)
        parent_id = project_ref['parent_id']
        parents_list = self.list_project_parents(parent_id)
        parent_ref = self.get_project(parent_id)
        parents_list.append(parent_ref)
        for ref in parents_list:
            if not ref.get('enabled', True):
                raise exception.ValidationError(message=_('cannot create a project in a branch containing a disabled project: %s') % ref['id'])
        self._assert_max_hierarchy_depth(project_ref.get('parent_id'), parents_list)