from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class StructuredDeploymentWithStrictInputParseTest(common.HeatTestCase):
    scenarios = SCENARIOS

    def test_parse(self):
        self.parse = sc.StructuredDeployment.parse
        if 'missing_input' not in self.shortDescription():
            self.assertEqual(self.result, self.parse(self.inputs, self.input_key, self.config, check_input_val='STRICT'))
        else:
            self.assertRaises(exception.UserParameterMissing, self.parse, self.inputs, self.input_key, self.config, check_input_val='STRICT')