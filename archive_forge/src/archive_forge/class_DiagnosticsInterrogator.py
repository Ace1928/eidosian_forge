from troveclient import base
class DiagnosticsInterrogator(base.ManagerWithFind):
    """Manager class for Interrogator resource."""
    resource_class = Diagnostics

    def get(self, instance):
        """Get the diagnostics of the guest on the instance."""
        return self._get('/mgmt/instances/%s/diagnostics' % base.getid(instance), 'diagnostics')

    def list(self):
        pass