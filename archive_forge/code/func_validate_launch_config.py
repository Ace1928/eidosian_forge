from unittest import mock
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def validate_launch_config(self, stack, lc_name='LaunchConfig'):
    conf = stack[lc_name]
    self.assertIsNone(conf.validate())
    scheduler.TaskRunner(conf.create)()
    self.assertEqual((conf.CREATE, conf.COMPLETE), conf.state)
    self.assertIsNotNone(conf.properties['BlockDeviceMappings'])