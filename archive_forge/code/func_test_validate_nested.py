from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def test_validate_nested(self):
    arglist = ['-t', self.template_path, '--show-nested']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    args = dict(self.defaults)
    args['show_nested'] = True
    self.stack_client.validate.assert_called_once_with(**args)
    self.assertEqual([], columns)
    self.assertEqual([], data)