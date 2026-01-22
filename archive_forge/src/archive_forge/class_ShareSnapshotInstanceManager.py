from manilaclient import api_versions
from manilaclient import base
class ShareSnapshotInstanceManager(base.ManagerWithFind):
    """Manage :class:`SnapshotInstances` resources."""
    resource_class = ShareSnapshotInstance

    @api_versions.wraps('2.19')
    def get(self, instance):
        """Get a snapshot instance.

        :param instance: either snapshot instance object or text with its ID.
        :rtype: :class:`ShareSnapshotInstance`
        """
        snapshot_instance_id = base.getid(instance)
        return self._get('/snapshot-instances/%s' % snapshot_instance_id, 'snapshot_instance')

    @api_versions.wraps('2.19')
    def list(self, detailed=False, snapshot=None, search_opts=None):
        """List all snapshot instances."""
        if detailed:
            url = '/snapshot-instances/detail'
        else:
            url = '/snapshot-instances'
        if snapshot:
            url += '?snapshot_id=%s' % base.getid(snapshot)
        return self._list(url, 'snapshot_instances')

    @api_versions.wraps('2.19')
    def reset_state(self, instance, state):
        """Reset the 'status' attr of the snapshot instance.

        :param instance: either snapshot instance object or its UUID.
        :param state: state to set the snapshot instance's 'status' attr to.
        """
        return self._action('reset_status', instance, {'status': state})

    def _action(self, action, instance, info=None, **kwargs):
        """Perform a snapshot instance 'action'."""
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/snapshot-instances/%s/action' % base.getid(instance)
        return self.api.client.post(url, body=body)