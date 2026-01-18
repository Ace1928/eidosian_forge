import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(service_objects.Service, 'get_all')
@mock.patch.object(service_utils, 'format_service')
def test_service_get_all(self, mock_format_service, mock_get_all):
    mock_get_all.return_value = [mock.Mock()]
    mock_format_service.return_value = mock.Mock()
    self.assertEqual(1, len(self.eng.list_services(self.ctx)))
    self.assertTrue(mock_get_all.called)
    mock_format_service.assert_called_once_with(mock.ANY)