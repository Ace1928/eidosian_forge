import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_sg_create(self):
    self._create_sg('test')
    expected_args = ()
    expected_kwargs = {'name': 'test', 'policy': 'anti-affinity', 'rules': {'max_server_per_host': 8}}
    self.sg_mgr.create.assert_called_once_with(*expected_args, **expected_kwargs)