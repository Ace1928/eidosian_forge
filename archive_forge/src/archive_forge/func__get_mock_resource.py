from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _get_mock_resource(self):
    value = mock.MagicMock()
    value.id = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    return value