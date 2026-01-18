import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_resp_does_not_log_sensitive_info(self):
    self.logger = self.useFixture(fixtures.FakeLogger(format='%(message)s', level=logging.DEBUG, nuke_handlers=True))
    cs = cinderclient.client.HTTPClient('user', None, None, 'http://127.0.0.1:5000')
    resp = mock.Mock()
    resp.status_code = 200
    resp.headers = {'x-compute-request-id': 'req-f551871a-4950-4225-9b2c-29a14c8f075e'}
    auth_password = 'kk4qD6CpKFLyz9JD'
    body = {'connection_info': {'driver_volume_type': 'iscsi', 'data': {'auth_password': auth_password, 'target_discovered': False, 'encrypted': False, 'qos_specs': None, 'target_iqn': 'iqn.2010-10.org.openstack:volume-a2f33dcc-1bb7-45ba-b8fc-5b38179120f8', 'target_portal': '10.0.100.186:3260', 'volume_id': 'a2f33dcc-1bb7-45ba-b8fc-5b38179120f8', 'target_lun': 1, 'access_mode': 'rw', 'auth_username': 's4BfSfZ67Bo2mnpuFWY8', 'auth_method': 'CHAP'}}}
    resp.text = jsonutils.dumps(body)
    cs.http_log_debug = True
    cs.http_log_resp(resp)
    output = self.logger.output.split('\n')
    self.assertIn('***', output[1], output)
    self.assertNotIn(auth_password, output[1], output)