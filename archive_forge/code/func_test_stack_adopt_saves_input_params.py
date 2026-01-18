from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def test_stack_adopt_saves_input_params(self):
    cfg.CONF.set_override('enable_stack_adopt', True)
    cfg.CONF.set_override('convergence_engine', False)
    env = {'parameters': {'app_dbx': 'foo'}}
    input_params = {'parameters': {'app_dbx': 'bar'}}
    template, adopt_data = self._get_adopt_data_and_template(env)
    result = self._do_adopt('test_adopt_saves_inputs', template, input_params, adopt_data)
    stack = stack_object.Stack.get_by_id(self.ctx, result['stack_id'])
    self.assertEqual(template, stack.raw_template.template)
    self.assertEqual(input_params['parameters'], stack.raw_template.environment['parameters'])