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
def _assert_max_hierarchy_depth(self, project_id, parents_list=None):
    if parents_list is None:
        parents_list = self.list_project_parents(project_id)
    max_depth = CONF.max_project_tree_depth + 1
    limit_model = PROVIDERS.unified_limit_api.enforcement_model
    if limit_model.MAX_PROJECT_TREE_DEPTH is not None:
        max_depth = min(max_depth, limit_model.MAX_PROJECT_TREE_DEPTH + 1)
    if self._get_hierarchy_depth(parents_list) > max_depth:
        raise exception.ForbiddenNotSecurity(_('Max hierarchy depth reached for %s branch.') % project_id)