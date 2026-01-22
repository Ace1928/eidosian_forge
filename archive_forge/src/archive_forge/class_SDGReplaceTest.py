import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SDGReplaceTest(common.HeatTestCase):
    template = {'heat_template_version': '2013-05-23', 'resources': {'deploy_mysql': {'type': 'OS::Heat::SoftwareDeploymentGroup', 'properties': {'config': 'config_uuid', 'servers': {'server1': 'uuid1', 'server2': 'uuid2'}, 'input_values': {'foo': 'bar'}, 'name': '10_config'}}}}
    scenarios = [('1', dict(count=2, existing=['0', '1'], batch_size=1, pause_sec=0, tasks=2)), ('2', dict(count=4, existing=['0', '1'], batch_size=3, pause_sec=0, tasks=2)), ('3', dict(count=3, existing=['0', '1'], batch_size=2, pause_sec=0, tasks=2)), ('4', dict(count=2, existing=['0', '1', '2'], batch_size=2, pause_sec=0, tasks=1)), ('5', dict(count=2, existing=['0', '1'], batch_size=1, pause_sec=1, tasks=3))]

    def get_fake_nested_stack(self, names):
        nested_t = '\n        heat_template_version: 2015-04-30\n        description: Resource Group\n        resources:\n        '
        resource_snip = "\n        '%s':\n            type: SoftwareDeployment\n            properties:\n              foo: bar\n        "
        resources = [nested_t]
        for res_name in names:
            resources.extend([resource_snip % res_name])
        nested_t = ''.join(resources)
        return utils.parse_stack(template_format.parse(nested_t))

    def setUp(self):
        super(SDGReplaceTest, self).setUp()
        self.stack = utils.parse_stack(self.template)
        snip = self.stack.t.resource_definitions(self.stack)['deploy_mysql']
        self.group = sd.SoftwareDeploymentGroup('deploy_mysql', snip, self.stack)
        self.group.update_with_template = mock.Mock()
        self.group.check_update_complete = mock.Mock()

    def test_rolling_updates(self):
        self.group._nested = self.get_fake_nested_stack(self.existing)
        self.group.get_size = mock.Mock(return_value=self.count)
        tasks = self.group._replace(0, self.batch_size, self.pause_sec)
        self.assertEqual(self.tasks, len(tasks))