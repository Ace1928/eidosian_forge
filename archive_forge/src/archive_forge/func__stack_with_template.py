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
def _stack_with_template(self, template_def):
    self.ctx = utils.dummy_context()
    self.stack = parser.Stack(self.ctx, 'software_deploly_test_stack', template.Template(template_def))
    self.deployment = self.stack['deploy_mysql']