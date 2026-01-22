import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class GroupsResource(ks_flask.ResourceBase):
    collection_key = 'groups'
    member_key = 'group'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='identity_api', method='get_group')

    def get(self, group_id=None):
        if group_id is not None:
            return self._get_group(group_id)
        return self._list_groups()

    def _get_group(self, group_id):
        """Get a group reference.

        GET/HEAD /groups/{group_id}
        """
        ENFORCER.enforce_call(action='identity:get_group', build_target=_build_group_target_enforcement)
        return self.wrap_member(PROVIDERS.identity_api.get_group(group_id))

    def _list_groups(self):
        """List groups.

        GET/HEAD /groups
        """
        filters = ['domain_id', 'name']
        target = None
        if self.oslo_context.domain_id:
            target = {'group': {'domain_id': self.oslo_context.domain_id}}
        ENFORCER.enforce_call(action='identity:list_groups', filters=filters, target_attr=target)
        hints = self.build_driver_hints(filters)
        domain = self._get_domain_id_for_list_request()
        refs = PROVIDERS.identity_api.list_groups(domain_scope=domain, hints=hints)
        if self.oslo_context.domain_id:
            filtered_refs = []
            for ref in refs:
                if ref['domain_id'] == target['group']['domain_id']:
                    filtered_refs.append(ref)
            refs = filtered_refs
        return self.wrap_collection(refs, hints=hints)

    def post(self):
        """Create group.

        POST /groups
        """
        group = self.request_body_json.get('group', {})
        target = {'group': group}
        ENFORCER.enforce_call(action='identity:create_group', target_attr=target)
        validation.lazy_validate(schema.group_create, group)
        group = self._normalize_dict(group)
        group = self._normalize_domain_id(group)
        ref = PROVIDERS.identity_api.create_group(group, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, group_id):
        """Update group.

        PATCH /groups/{group_id}
        """
        ENFORCER.enforce_call(action='identity:update_group', build_target=_build_group_target_enforcement)
        group = self.request_body_json.get('group', {})
        validation.lazy_validate(schema.group_update, group)
        self._require_matching_id(group)
        ref = PROVIDERS.identity_api.update_group(group_id, group, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, group_id):
        """Delete group.

        DELETE /groups/{group_id}
        """
        ENFORCER.enforce_call(action='identity:delete_group')
        PROVIDERS.identity_api.delete_group(group_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)