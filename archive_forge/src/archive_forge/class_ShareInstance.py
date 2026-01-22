from openstack import exceptions
from openstack import resource
from openstack import utils
class ShareInstance(resource.Resource):
    resource_key = 'share_instance'
    resources_key = 'share_instances'
    base_path = '/share_instances'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True
    allow_head = False
    access_rules_status = resource.Body('access_rules_status', type=str)
    availability_zone = resource.Body('availability_zone', type=str)
    cast_rules_to_readonly = resource.Body('cast_rules_to_readonly', type=bool)
    created_at = resource.Body('created_at', type=str)
    host = resource.Body('host', type=str)
    progress = resource.Body('progress', type=str)
    replica_state = resource.Body('replica_state', type=str)
    share_id = resource.Body('share_id', type=str)
    share_network_id = resource.Body('share_network_id', type=str)
    share_server_id = resource.Body('share_server_id', type=str)
    status = resource.Body('status', type=str)

    def _action(self, session, body, action='patch', microversion=None):
        """Perform share instance actions given the message body"""
        url = utils.urljoin(self.base_path, self.id, 'action')
        headers = {'Accept': ''}
        extra_attrs = {}
        if microversion:
            extra_attrs['microversion'] = microversion
        else:
            extra_attrs['microversion'] = self._get_microversion(session, action=action)
        response = session.post(url, json=body, headers=headers, **extra_attrs)
        exceptions.raise_from_response(response)
        return response

    def reset_status(self, session, reset_status):
        """Reset share instance to given status"""
        body = {'reset_status': {'status': reset_status}}
        self._action(session, body)

    def force_delete(self, session):
        """Force delete share instance"""
        body = {'force_delete': None}
        self._action(session, body, action='delete')