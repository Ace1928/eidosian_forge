from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_node_processes_fails(self):
    ngt = self._init_ngt(self.t)
    plugin_mock = mock.MagicMock()
    plugin_mock.node_processes = {'HDFS': ['namenode', 'datanode', 'secondarynamenode'], 'JobFlow': ['oozie']}
    self.plugin_mgr.get_version_details.return_value = plugin_mock
    ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
    self.assertIn("resources.node-group.properties: Plugin vanilla doesn't support the following node processes: jobtracker. Allowed processes are: ", str(ex))
    self.assertIn('namenode', str(ex))
    self.assertIn('datanode', str(ex))
    self.assertIn('secondarynamenode', str(ex))
    self.assertIn('oozie', str(ex))