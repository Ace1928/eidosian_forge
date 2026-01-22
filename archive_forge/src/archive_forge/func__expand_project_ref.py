import functools
import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.resource import schema
from keystone.server import flask as ks_flask
def _expand_project_ref(self, ref):
    parents_as_list = self.query_filter_is_true('parents_as_list')
    parents_as_ids = self.query_filter_is_true('parents_as_ids')
    subtree_as_list = self.query_filter_is_true('subtree_as_list')
    subtree_as_ids = self.query_filter_is_true('subtree_as_ids')
    include_limits = self.query_filter_is_true('include_limits')
    if parents_as_list and parents_as_ids:
        msg = _('Cannot use parents_as_list and parents_as_ids query params at the same time.')
        raise exception.ValidationError(msg)
    if subtree_as_list and subtree_as_ids:
        msg = _('Cannot use subtree_as_list and subtree_as_ids query params at the same time.')
        raise exception.ValidationError(msg)
    if parents_as_list:
        parents = PROVIDERS.resource_api.list_project_parents(ref['id'], self.oslo_context.user_id, include_limits)
        ref['parents'] = [self.wrap_member(p) for p in parents]
    elif parents_as_ids:
        ref['parents'] = PROVIDERS.resource_api.get_project_parents_as_ids(ref)
    if subtree_as_list:
        subtree = PROVIDERS.resource_api.list_projects_in_subtree(ref['id'], self.oslo_context.user_id, include_limits)
        ref['subtree'] = [self.wrap_member(p) for p in subtree]
    elif subtree_as_ids:
        ref['subtree'] = PROVIDERS.resource_api.get_projects_in_subtree_as_ids(ref['id'])