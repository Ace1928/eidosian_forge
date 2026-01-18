import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_role_handle_create_with_domain(self):
    self._test_handle_create(domain='d_test')