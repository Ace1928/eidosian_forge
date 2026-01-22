from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_qos_full(id, base_uri, tenant_id, name=None, specs=None):
    if not name:
        name = 'fake-name'
    if not specs:
        specs = {}
    return {'qos_specs': {'id': id, 'name': name, 'consumer': 'back-end', 'specs': specs}, 'links': {'href': _bookmark_href(base_uri, tenant_id, id), 'rel': 'bookmark'}}