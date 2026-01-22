from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeTypeAccessManager(base.ManagerWithFind):
    """
    Manage :class:`VolumeTypeAccess` resources.
    """
    resource_class = VolumeTypeAccess

    def list(self, volume_type):
        return self._list('/types/%s/os-volume-type-access' % base.getid(volume_type), 'volume_type_access')

    def add_project_access(self, volume_type, project):
        """Add a project to the given volume type access list."""
        info = {'project': project}
        return self._action('addProjectAccess', volume_type, info)

    def remove_project_access(self, volume_type, project):
        """Remove a project from the given volume type access list."""
        info = {'project': project}
        return self._action('removeProjectAccess', volume_type, info)

    def _action(self, action, volume_type, info, **kwargs):
        """Perform a volume type action."""
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/types/%s/action' % base.getid(volume_type)
        resp, body = self.api.client.post(url, body=body)
        return common_base.TupleWithMeta((resp, body), resp)