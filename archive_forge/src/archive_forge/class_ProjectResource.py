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
class ProjectResource(ks_flask.ResourceBase):
    collection_key = 'projects'
    member_key = 'project'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='resource_api', method='get_project')

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

    def _get_project(self, project_id):
        """Get project.

        GET/HEAD /v3/projects/{project_id}
        """
        ENFORCER.enforce_call(action='identity:get_project', build_target=_build_project_target_enforcement)
        project = PROVIDERS.resource_api.get_project(project_id)
        self._expand_project_ref(project)
        return self.wrap_member(project)

    def _list_projects(self):
        """List projects.

        GET/HEAD /v3/projects
        """
        filters = ('domain_id', 'enabled', 'name', 'parent_id', 'is_domain')
        target = None
        if self.oslo_context.domain_id:
            target = {'domain_id': self.oslo_context.domain_id}
        ENFORCER.enforce_call(action='identity:list_projects', filters=filters, target_attr=target)
        hints = self.build_driver_hints(filters)
        if 'is_domain' not in flask.request.args:
            hints.add_filter('is_domain', '0')
        tag_params = ['tags', 'tags-any', 'not-tags', 'not-tags-any']
        for t in tag_params:
            if t in flask.request.args:
                hints.add_filter(t, flask.request.args[t])
        refs = PROVIDERS.resource_api.list_projects(hints=hints)
        if self.oslo_context.domain_id:
            domain_id = self.oslo_context.domain_id
            filtered_refs = [ref for ref in refs if ref['domain_id'] == domain_id]
        else:
            filtered_refs = refs
        return self.wrap_collection(filtered_refs, hints=hints)

    def get(self, project_id=None):
        """Get project or list projects.

        GET/HEAD /v3/projects
        GET/HEAD /v3/projects/{project_id}
        """
        if project_id is not None:
            return self._get_project(project_id)
        else:
            return self._list_projects()

    def post(self):
        """Create project.

        POST /v3/projects
        """
        project = self.request_body_json.get('project', {})
        target = {'project': project}
        ENFORCER.enforce_call(action='identity:create_project', target_attr=target)
        validation.lazy_validate(schema.project_create, project)
        project = self._assign_unique_id(project)
        if not project.get('is_domain'):
            project = self._normalize_domain_id(project)
        if not project.get('parent_id'):
            project['parent_id'] = project.get('domain_id')
        project = self._normalize_dict(project)
        try:
            ref = PROVIDERS.resource_api.create_project(project['id'], project, initiator=self.audit_initiator)
        except (exception.DomainNotFound, exception.ProjectNotFound) as e:
            raise exception.ValidationError(e)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, project_id):
        """Update project.

        PATCH /v3/projects/{project_id}
        """
        ENFORCER.enforce_call(action='identity:update_project', build_target=_build_project_target_enforcement)
        project = self.request_body_json.get('project', {})
        validation.lazy_validate(schema.project_update, project)
        self._require_matching_id(project)
        ref = PROVIDERS.resource_api.update_project(project_id, project, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, project_id):
        """Delete project.

        DELETE /v3/projects/{project_id}
        """
        ENFORCER.enforce_call(action='identity:delete_project', build_target=_build_project_target_enforcement)
        PROVIDERS.resource_api.delete_project(project_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)