from openstack import resource
from openstack import utils
class ShareGroupSnapshot(resource.Resource):
    resource_key = 'share_group_snapshot'
    resources_key = 'share_group_snapshots'
    base_path = '/share-group-snapshots'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('project_id', 'all_tenants', 'name', 'description', 'status', 'share_group_id', 'limit', 'offset', 'sort_key', 'sort_dir')
    project_id = resource.Body('project_id', type=str)
    status = resource.Body('status', type=str)
    share_group_id = resource.Body('share_group_id', type=str)
    description = resource.Body('description', type=str)
    created_at = resource.Body('created_at', type=str)
    members = resource.Body('members', type=str)
    size = resource.Body('size', type=int)
    share_protocol = resource.Body('share_proto', type=str)

    def _action(self, session, body, action='patch', microversion=None):
        """Perform ShareGroupSnapshot actions given the message body."""
        url = utils.urljoin(self.base_path, self.id, 'action')
        headers = {'Accept': ''}
        microversion = microversion or self._get_microversion(session, action=action)
        extra_attrs = {'microversion': microversion}
        session.post(url, json=body, headers=headers, **extra_attrs)

    def reset_status(self, session, status):
        body = {'reset_status': {'status': status}}
        self._action(session, body)

    def get_members(self, session, microversion=None):
        url = utils.urljoin(self.base_path, self.id, 'members')
        microversion = microversion or self._get_microversion(session, action='list')
        headers = {'Accept': ''}
        response = session.get(url, headers=headers, microversion=microversion)
        return response.json()