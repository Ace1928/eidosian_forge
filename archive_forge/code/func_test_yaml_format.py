import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
def test_yaml_format(self):
    self.cmd = ShowYaml(self.app, None)
    parsed_args = self.check_parser(self.cmd, [], [])
    expected = yaml.safe_dump(dict(zip(columns, data)), default_flow_style=False)
    self.cmd.run(parsed_args)
    self.assertEqual(expected, self.app.stdout.make_string())