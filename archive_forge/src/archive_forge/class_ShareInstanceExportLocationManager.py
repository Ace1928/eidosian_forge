from manilaclient import api_versions
from manilaclient import base
class ShareInstanceExportLocationManager(base.ManagerWithFind):
    """Manage :class:`ShareInstanceExportLocation` resources."""
    resource_class = ShareInstanceExportLocation

    @api_versions.wraps('2.9')
    def list(self, share_instance, search_opts=None):
        """List all share export locations."""
        share_instance_id = base.getid(share_instance)
        return self._list('/share_instances/%s/export_locations' % share_instance_id, 'export_locations')

    @api_versions.wraps('2.9')
    def get(self, share_instance, export_location):
        """Get a share export location."""
        share_instance_id = base.getid(share_instance)
        export_location_id = base.getid(export_location)
        return self._get('/share_instances/%(share_instance_id)s/export_locations/%(export_location_id)s' % {'share_instance_id': share_instance_id, 'export_location_id': export_location_id}, 'export_location')