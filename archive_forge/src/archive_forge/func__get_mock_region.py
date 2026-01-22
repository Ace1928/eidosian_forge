from unittest import mock
from urllib import parse
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import region
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _get_mock_region(self):
    value = mock.MagicMock()
    region_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    value.id = region_id
    return value