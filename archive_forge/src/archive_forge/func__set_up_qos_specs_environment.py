from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _set_up_qos_specs_environment(self):
    self.qos_specs.create.return_value = self.value
    self.my_qos_specs.handle_create()