from openstack import exceptions
from openstack import resource
from openstack import utils
class GroupSnapshot(resource.Resource):
    resource_key = 'group_snapshot'
    resources_key = 'group_snapshots'
    base_path = '/group_snapshots'
    allow_fetch = True
    allow_create = True
    allow_delete = True
    allow_commit = False
    allow_list = True
    _query_mapping = resource.QueryParameters('limit', 'marker', 'offset', 'sort_dir', 'sort_key', 'sort', all_projects='all_tenants')
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    group_id = resource.Body('group_id')
    group_type_id = resource.Body('group_type_id')
    id = resource.Body('id')
    name = resource.Body('name')
    project_id = resource.Body('project_id')
    status = resource.Body('status')
    _max_microversion = '3.29'

    def _action(self, session, body, microversion=None):
        """Preform aggregate actions given the message body."""
        url = utils.urljoin(self.base_path, self.id, 'action')
        headers = {'Accept': ''}
        if microversion is None:
            if session.default_microversion:
                microversion = session.default_microversion
            else:
                microversion = utils.maximum_supported_microversion(session, self._max_microversion)
        response = session.post(url, json=body, headers=headers, microversion=microversion)
        exceptions.raise_from_response(response)
        return response

    def reset_state(self, session, state):
        """Resets the status for a group snapshot."""
        body = {'reset_status': {'status': state}}
        return self._action(session, body)