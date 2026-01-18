import copy
from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster_template
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_invalid_volume_driver(self):
    props = self.t['resources']['test_cluster_template']['properties']
    props['volume_driver'] = 'cinder'
    stack = utils.parse_stack(self.t)
    msg = "Volume driver type cinder is not supported by COE:mesos, expecting a ['rexray'] volume driver."
    ex = self.assertRaises(exception.StackValidationFailed, stack['test_cluster_template'].validate)
    self.assertEqual(msg, str(ex))