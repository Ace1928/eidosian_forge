from unittest import mock
import ddt
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import security_services
def test_update_by_object(self):
    security_service = self._FakeSecurityService()
    values = {'user': 'fake_user'}
    with mock.patch.object(self.manager, '_update', fakes.fake_update):
        result = self.manager.update(security_service, **values)
        self.assertEqual(result['url'], security_services.RESOURCE_PATH % security_service.id)
        self.assertEqual(result['resp_key'], security_services.RESOURCE_NAME)
        self.assertEqual(result['body'][security_services.RESOURCE_NAME], values)