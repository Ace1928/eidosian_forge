from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class ZoneImportsController(V2Controller):

    def create(self, zone_file_contents):
        return self._post('/zones/tasks/imports', data=zone_file_contents, headers={'Content-Type': 'text/dns'})

    def get_import_record(self, zone_import_id):
        return self._get(f'/zones/tasks/imports/{zone_import_id}')

    def list(self):
        return self._get('/zones/tasks/imports')

    def delete(self, zone_import_id):
        return self._delete(f'/zones/tasks/imports/{zone_import_id}')