from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_backup_full(id, base_uri, tenant_id):
    return {'id': id, 'name': 'backup', 'description': 'nightly backup', 'volume_id': '712f4980-5ac1-41e5-9383-390aa7c9f58b', 'container': 'volumebackups', 'object_count': 220, 'size': 10, 'availability_zone': 'az1', 'created_at': '2013-04-12T08:16:37.000000', 'status': 'available', 'links': [{'href': _self_href(base_uri, tenant_id, id), 'rel': 'self'}, {'href': _bookmark_href(base_uri, tenant_id, id), 'rel': 'bookmark'}]}