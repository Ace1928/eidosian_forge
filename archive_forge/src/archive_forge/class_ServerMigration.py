from openstack import exceptions
from openstack import resource
from openstack import utils
class ServerMigration(resource.Resource):
    resource_key = 'migration'
    resources_key = 'migrations'
    base_path = '/servers/%(server_id)s/migrations'
    allow_fetch = True
    allow_list = True
    allow_delete = True
    server_id = resource.URI('server_id')
    created_at = resource.Body('created_at')
    dest_host = resource.Body('dest_host')
    dest_compute = resource.Body('dest_compute')
    dest_node = resource.Body('dest_node')
    disk_processed_bytes = resource.Body('disk_processed_bytes')
    disk_remaining_bytes = resource.Body('disk_remaining_bytes')
    disk_total_bytes = resource.Body('disk_total_bytes')
    memory_processed_bytes = resource.Body('memory_processed_bytes')
    memory_remaining_bytes = resource.Body('memory_remaining_bytes')
    memory_total_bytes = resource.Body('memory_total_bytes')
    project_id = resource.Body('project_id')
    server_uuid = resource.Body('server_uuid')
    source_compute = resource.Body('source_compute')
    source_node = resource.Body('source_node')
    status = resource.Body('status')
    updated_at = resource.Body('updated_at')
    user_id = resource.Body('user_id')
    uuid = resource.Body('uuid', alternate_id=True)
    _max_microversion = '2.80'

    def _action(self, session, body):
        """Preform server migration actions given the message body."""
        session = self._get_session(session)
        microversion = self._get_microversion(session, action='list')
        url = utils.urljoin(self.base_path % {'server_id': self.server_id}, self.id, 'action')
        response = session.post(url, microversion=microversion, json=body)
        exceptions.raise_from_response(response)
        return response

    def force_complete(self, session):
        """Force on-going live migration to complete."""
        body = {'force_complete': None}
        self._action(session, body)