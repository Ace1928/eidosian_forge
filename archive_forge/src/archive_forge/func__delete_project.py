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
def _delete_project(self, project, initiator=None, cascade=False):
    ro_opt.check_immutable_delete(resource_ref=project, resource_type='project', resource_id=project['id'])
    project_id = project['id']
    if project['is_domain'] and project['enabled']:
        raise exception.ValidationError(message=_('cannot delete an enabled project acting as a domain. Please disable the project %s first.') % project.get('id'))
    if not self.is_leaf_project(project_id) and (not cascade):
        raise exception.ForbiddenNotSecurity(_('Cannot delete the project %s since it is not a leaf in the hierarchy. Use the cascade option if you want to delete a whole subtree.') % project_id)
    if cascade:
        subtree_list = self.list_projects_in_subtree(project_id)
        subtree_list.reverse()
        if not self._check_whole_subtree_is_disabled(project_id, subtree_list=subtree_list):
            raise exception.ForbiddenNotSecurity(_('Cannot delete project %(project_id)s since its subtree contains enabled projects.') % {'project_id': project_id})
        project_list = subtree_list + [project]
        projects_ids = [x['id'] for x in project_list]
        ret = self.driver.delete_projects_from_ids(projects_ids)
        for prj in project_list:
            self._post_delete_cleanup_project(prj['id'], prj, initiator)
    else:
        ret = self.driver.delete_project(project_id)
        self._post_delete_cleanup_project(project_id, project, initiator)
    reason = 'The token cache is being invalidate because project %(project_id)s was deleted. Authorization will be recalculated and enforced accordingly the next time users authenticate or validate a token.' % {'project_id': project_id}
    notifications.invalidate_token_cache_notification(reason)
    return ret