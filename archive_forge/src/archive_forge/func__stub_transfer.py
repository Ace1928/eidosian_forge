from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_transfer(id, base_uri, tenant_id):
    return {'id': id, 'name': 'transfer', 'volume_id': '8c05f861-6052-4df6-b3e0-0aebfbe686cc', 'links': [{'href': _self_href(base_uri, tenant_id, id), 'rel': 'self'}, {'href': _bookmark_href(base_uri, tenant_id, id), 'rel': 'bookmark'}]}