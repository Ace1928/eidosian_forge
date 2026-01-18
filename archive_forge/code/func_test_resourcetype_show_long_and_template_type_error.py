from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource_type
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resource_types
def test_resourcetype_show_long_and_template_type_error(self):
    arglist = ['OS::Heat::None', '--template-type', 'cfn', '--long']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)