import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
class TestQosShow(TestQos):
    qos_spec = volume_fakes.create_one_qos()
    qos_association = volume_fakes.create_one_qos_association()
    columns = ('associations', 'consumer', 'id', 'name', 'properties')
    data = (format_columns.ListColumn([qos_association.name]), qos_spec.consumer, qos_spec.id, qos_spec.name, format_columns.DictColumn(qos_spec.specs))

    def setUp(self):
        super(TestQosShow, self).setUp()
        self.qos_mock.get.return_value = self.qos_spec
        self.qos_mock.get_associations.return_value = [self.qos_association]
        self.cmd = qos_specs.ShowQos(self.app, None)

    def test_qos_show(self):
        arglist = [self.qos_spec.id]
        verifylist = [('qos_spec', self.qos_spec.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.qos_mock.get.assert_called_with(self.qos_spec.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, tuple(data))