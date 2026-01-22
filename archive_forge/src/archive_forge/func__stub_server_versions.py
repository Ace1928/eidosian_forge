from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_server_versions():
    return [{'status': 'SUPPORTED', 'updated': '2015-07-30T11:33:21Z', 'links': [{'href': 'http://docs.openstack.org/', 'type': 'text/html', 'rel': 'describedby'}, {'href': 'http://localhost:8776/v1/', 'rel': 'self'}], 'min_version': '', 'version': '', 'id': 'v1.0'}, {'status': 'SUPPORTED', 'updated': '2015-09-30T11:33:21Z', 'links': [{'href': 'http://docs.openstack.org/', 'type': 'text/html', 'rel': 'describedby'}, {'href': 'http://localhost:8776/v2/', 'rel': 'self'}], 'min_version': '', 'version': '', 'id': 'v2.0'}, {'status': 'CURRENT', 'updated': '2016-04-01T11:33:21Z', 'links': [{'href': 'http://docs.openstack.org/', 'type': 'text/html', 'rel': 'describedby'}, {'href': 'http://localhost:8776/v3/', 'rel': 'self'}], 'min_version': '3.0', 'version': '3.1', 'id': 'v3.0'}]