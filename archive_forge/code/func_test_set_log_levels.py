from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_set_log_levels(self):
    expected = {'level': 'debug', 'binary': 'cinder-api', 'server': 'host1', 'prefix': 'sqlalchemy.'}
    cs = fakes.FakeClient(version_header='3.32')
    cs.services.set_log_levels(expected['level'], expected['binary'], expected['server'], expected['prefix'])
    cs.assert_called('PUT', '/os-services/set-log', body=expected)