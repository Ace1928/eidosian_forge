from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_backup(id, base_uri, tenant_id):
    return {'id': id, 'name': 'backup', 'links': [{'href': _self_href(base_uri, tenant_id, id), 'rel': 'self'}, {'href': _bookmark_href(base_uri, tenant_id, id), 'rel': 'bookmark'}]}