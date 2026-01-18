import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_user(self):
    self.patchobject(keystone.KeystoneClientPlugin, 'get_user_id', return_value='user_A')
    self.patchobject(nova.NovaClientPlugin, 'get_max_microversion', return_value='2.1')
    self._test_validate(user='user_A')