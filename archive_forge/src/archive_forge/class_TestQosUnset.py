import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
class TestQosUnset(TestQos):
    qos_spec = volume_fakes.create_one_qos()

    def setUp(self):
        super(TestQosUnset, self).setUp()
        self.qos_mock.get.return_value = self.qos_spec
        self.cmd = qos_specs.UnsetQos(self.app, None)

    def test_qos_unset_with_properties(self):
        arglist = ['--property', 'iops', '--property', 'foo', self.qos_spec.id]
        verifylist = [('property', ['iops', 'foo']), ('qos_spec', self.qos_spec.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.qos_mock.unset_keys.assert_called_with(self.qos_spec.id, ['iops', 'foo'])
        self.assertIsNone(result)