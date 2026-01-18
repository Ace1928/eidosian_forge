from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def test_stack_adopt_disabled(self):
    cfg.CONF.set_override('enable_stack_adopt', False)
    env = {'parameters': {'app_dbx': 'test'}}
    template, adopt_data = self._get_adopt_data_and_template(env)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.create_stack, self.ctx, 'test_adopt_stack_disabled', template, {}, None, {'adopt_stack_data': str(adopt_data)})
    self.assertEqual(exception.NotSupported, ex.exc_info[0])
    self.assertIn('Stack Adopt', str(ex.exc_info[1]))