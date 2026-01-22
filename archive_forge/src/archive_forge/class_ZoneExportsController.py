from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class ZoneExportsController(V2Controller):

    def create(self, zone):
        zone_id = v2_utils.resolve_by_name(self.client.zones.list, zone)
        return self._post(f'/zones/{zone_id}/tasks/export')

    def get_export_record(self, zone_export_id):
        return self._get(f'/zones/tasks/exports/{zone_export_id}')

    def list(self):
        return self._get('/zones/tasks/exports')

    def delete(self, zone_export_id):
        return self._delete(f'/zones/tasks/exports/{zone_export_id}')

    def get_export(self, zone_export_id):
        return self._get(f'/zones/tasks/exports/{zone_export_id}/export', headers={'Accept': 'text/dns'})