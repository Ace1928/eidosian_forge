import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _invalidate_token_cache(self, role_id, group_id, user_id, project_id, domain_id):
    if group_id:
        actor_type = 'group'
        actor_id = group_id
    elif user_id:
        actor_type = 'user'
        actor_id = user_id
    if domain_id:
        target_type = 'domain'
        target_id = domain_id
    elif project_id:
        target_type = 'project'
        target_id = project_id
    reason = 'Invalidating the token cache because role %(role_id)s was removed from %(actor_type)s %(actor_id)s on %(target_type)s %(target_id)s.' % {'role_id': role_id, 'actor_type': actor_type, 'actor_id': actor_id, 'target_type': target_type, 'target_id': target_id}
    notifications.invalidate_token_cache_notification(reason)